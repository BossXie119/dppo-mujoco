import os
import io
import zmq
import time
import torch
import logger
import pickle
import random
import scipy.signal

import gymnasium as gym
import numpy as np

from env import make_env
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Array
from pathlib import Path
from itertools import count
from multiprocessing import Process
from argparse import ArgumentParser
from model import Agent
from mem_pool import MemPool
from pyarrow import serialize

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='dppo-mujoco', help='The RL algorithm')
parser.add_argument('--exp', type=str, default='rate-return-index', help='The explanation of this experiment')
parser.add_argument('--env_id', type=str, default='HalfCheetah-v4', help='The game environment')
parser.add_argument('--num_actor', type=str, default='actor1', help='The game environment')
parser.add_argument('--num_steps', type=int, default=1000000, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
parser.add_argument('--torch_deterministic', type=bool, default=True, help='if toggled, `torch.backends.cudnn.deterministic=False')
parser.add_argument('--max_steps_per_update', type=int, default=2048,
                    help='The maximum number of steps between each update')
parser.add_argument('--data_path', type=str, default='DATA1',help='Directory to save logging data')
parser.add_argument('--pth_path', type=str, default='/PTH',help='Directory model parameters and config file')
parser.add_argument('--num_saved_pth', type=int, default=5, help='Number of recent checkpoint files to be saved')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--num_envs', type=int, default=1, help='the number of parallel game environments')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='the lambda for the general advantage estimation')
parser.add_argument('--index', type=int, default=1, help='the index of actors')

def run_one_agent(args, actor_status):
    # Connect to learner
    context = zmq.Context()
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    if args.index == 1:
        run_name = f"{args.env_id}__{args.alg}__{args.num_actor}__{args.seed}__{int(time.time())}"
        writering = SummaryWriter(f"ACTOR3/runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    #device = torch.device("cpu")
    model_id = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Init envs and agent
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma) for i in range(args.num_envs)],
    )
    agent = Agent(envs).to(device)

    while True:
        new_weights, model_id = find_new_weights(model_id, args.pth_path)
        if new_weights is not None:
            equal = all(torch.equal(agent.state_dict()[key], new_weights[key]) for key in new_weights.keys())
            print(equal)
            agent.load_state_dict(new_weights)
            break
            
    # Set logging path
    if args.index ==1:
        logger.configure(str(args.log_path))

    # A list to store raw transitions within an episode
    transitions = []  
    # A pool to store prepared training data
    mem_pool = MemPool()  

    episode_rewards = [0.0]
    episode_lengths = [0]
    num_episodes = 0
    mean_10ep_reward = 0
    mean_10ep_length = 0
    send_time_start = time.time()

    state, _ = envs.reset(seed=(args.index + args.seed))
    state = torch.Tensor(state).to(device)

    for step in range(args.num_steps):
        # Sample action
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(state)
        # Next step
        next_state, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        # Record transitionstate
        transitions.append((np.squeeze(state.numpy()), np.squeeze(action.numpy()), reward.item(), logprob.item(), value.item(), next_state, done.item()))
        episode_rewards[-1] += reward
        episode_lengths[-1] += 1

        state = torch.Tensor(next_state).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    if args.index == 1 and step % 10 == 0:
                        writering.add_scalar("charts/episodic_return", info["episode"]["r"], step)
                        writering.add_scalar("charts/episodic_length", info["episode"]["l"], step)
                        logger.record_tabular("Epoch", step)
                        logger.record_tabular("AverageEpRet", info["episode"]["r"][0])
                        logger.dump_tabular()
                    else:
                        print(f"global_step={step}, episodic_return={info['episode']['r']}")
                        print(f"global_step={step}, episodic_length={info['episode']['l']}")

        is_terminal = done or episode_lengths[-1] >= args.max_episode_length > 0
        if is_terminal or len(mem_pool) + len(transitions) >= args.max_steps_per_update:
            # Current episode is terminated or a trajectory of enough training data is collected
            data = prepare_training_data(agent, transitions, args.gamma, args.gae_lambda)
            transitions.clear()
            mem_pool.push(data)

            if is_terminal:
                # Log information at the end of episode
                num_episodes = len(episode_rewards)
                mean_10ep_reward = round(np.mean(episode_rewards[-10:]), 2)
                mean_10ep_length = round(np.mean(episode_lengths[-10:]), 2)
                # print(episode_rewards[-1])
                episode_rewards.append(0.0)
                episode_lengths.append(0)

                state, _ = envs.reset(seed=(args.index + args.seed))
                state = torch.Tensor(state).to(device)
        
        if len(mem_pool) >= args.max_steps_per_update:
            # Send training data after enough training data (>= 'arg.max_steps_per_update') is collected
            post_processed_data = post_process_training_data(training_data = mem_pool.sample())
            # post_processed_data = mem_pool.sample()
            # Serialize a general Python sequence for transient storage and transport
            socket.send(serialize(post_processed_data).to_buffer())
            socket.recv()
            mem_pool.clear()

            send_data_interval = time.time() - send_time_start
            send_time_start = time.time()

            print(f"current_iteration={(step + 1) // args.max_steps_per_update}, collection_time={send_data_interval}")

            # if num_episodes > 0:
            #     # Log information
            #     logger.record_tabular("iteration", (step + 1) // args.max_steps_per_update)
            #     logger.record_tabular("steps", step)
            #     logger.record_tabular("episodes", len(episode_rewards))
            #     logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
            #     logger.record_tabular("mean 10 episode length", mean_10ep_length)
            #     logger.record_tabular("send data fps", args.max_steps_per_update // send_data_interval)
            #     logger.record_tabular("send data interval", send_data_interval)
            #     logger.dump_tabular()

        # Get new weight
        new_weights, model_id = find_new_weights(model_id, args.pth_path)
        
        if new_weights is not None:
            # equal = all(torch.equal(agent.state_dict()[key], new_weights[key]) for key in new_weights.keys())
            # print(equal)
            agent.load_state_dict(new_weights)
    print("all steps done")
    actor_status[0] = 1

def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.pth_path / f'{model_id}.{args.alg}.{args.env_id}.pth', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_pth:
                    os.remove(args.pth_path / f'{model_id - args.num_saved_pth}.{args.alg}.{args.env_id}.pth')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # actor finished works
                return

            # For not cpu-intensive
            time.sleep(1)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def find_new_weights(current_model_id, pth_path):
    try:
        pth_files = sorted(os.listdir(pth_path), key=lambda p: int(p.split('.')[0]))
        latest_file = pth_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(pth_path / latest_file, 'rb') as f:
                    new_weights = CPU_Unpickler(f).load()
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id
    
def prepare_training_data(agent, trajectory, gamma, gae_lambda):
    states, actions, rewards, logprob, values = [np.array(i) for i in list(zip(*trajectory))[:5]]
    next_state = trajectory[-1][5]
    done = trajectory[-1][6]

    last_val = (1 - done) * agent.get_value(torch.Tensor(next_state)).item()
    values = np.append(values, last_val)
    rewards = np.append(rewards, last_val)
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    advantages = discount_cumulative_sum(deltas, gamma * gae_lambda)
    # returns = discount_cumulative_sum(rewards, gamma)[:-1]
    returns = advantages + values[:-1]

    return {
        'state': states,
        'action': actions,
        'return': returns,
        'act_prob': logprob,
        'advantage': advantages,
        'value': values[:-1]
    }

def discount_cumulative_sum(x, discount):
    """
    Magic from RLLab for computing discounted cumulative sums of vectors.
    :param x: [x0, x1, x2]
    :param discount: Discount coefficient
    :return: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def post_process_training_data(training_data):
    advantage = training_data['advantage']

    mean = np.sum(advantage) / len(advantage)
    std = np.sqrt(np.sum((advantage - mean) ** 2) / len(advantage))
    training_data['advantage'] = (advantage - mean) / (std + 1e-8)

    return training_data

def main():
    # Parse input parameters
    args, _ = parser.parse_known_args()

    args.index = int(args.index)
    args.data_path = Path(args.env_id)
    args.pth_path = Path(args.pth_path)
    path = args.num_actor + "-" + str(args.seed)
    args.log_path = args.data_path / args.num_actor / path
    args.data_path.mkdir(exist_ok=True)
    args.pth_path.mkdir(exist_ok=True)
    args.log_path.mkdir(parents=True, exist_ok=True)

    # Running status of actor
    actor_status = Array('i', [0])

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    one_agent = Process(target=run_one_agent, args=(args, actor_status))
    one_agent.start()

    one_agent.join()
    subscriber.join()


if __name__ == '__main__':
    main()


