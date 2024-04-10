import zmq
import time
import torch
import pickle
import random
import multiprocessing

import numpy as np

from env import make_env

import torch.nn as nn
from model import Agent
import gymnasium as gym
import torch.optim as optim
from argparse import ArgumentParser
from itertools import count
from multiprocessing import Process
from pyarrow import deserialize
from torch.utils.tensorboard import SummaryWriter
from mem_pool import MemPoolManager, MultiprocessingMemPool

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='dppo-mujoco', help='The RL algorithm')
parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
parser.add_argument('--cuda', type=bool, default=True, help='if toggled, cuda will be enabled by default')
parser.add_argument('--torch_deterministic', type=bool, default=True, help='if toggled, `torch.backends.cudnn.deterministic=False')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
parser.add_argument('--env_id', type=str, default='HalfCheetah-v4', help='The game environment')
parser.add_argument('--num_envs', type=int, default=1, help='the number of parallel game environments')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='the learning rate of the optimizer')
parser.add_argument('--clip_coef', type=float, default=0.2, help='the surrogate clipping coefficient')
parser.add_argument('--clip_vloss', type=bool, default=True, help='Toggles whether or not to use a clipped loss for the value function, as per the paper')
parser.add_argument('--ent_coef', type=float, default=0.0, help='coefficient of the entropy')
parser.add_argument('--vf_coef', type=float, default=0.5, help='coefficient of the value function')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='the maximum norm for the gradient clipping')
parser.add_argument('--pool_size', type=int, default=2048, help='The max length of data pool')
parser.add_argument('--update_epochs', type=int, default=10, help='the K epochs to update the policy')
parser.add_argument('--training_freq', type=int, default=1,
                    help='How many receptions of new data are between each training, '
                         'which can be fractional to represent more than one training per reception')
parser.add_argument('--batch_size', type=int, default=2048, help='The batch size for training')
parser.add_argument('--record_throughput_interval', type=int, default=10,
                    help='The time interval between each throughput record')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
parser.add_argument('--num_actors', type=int, default=1, help='The number of actors')
parser.add_argument('--update_step', type=int, default=0, help='The number of update_step')
parser.add_argument('--num_steps', type=int, default=1000000, help='The number of num_steps')
parser.add_argument('--num_iterations', type=int, default=0, help='The number of num_iterations')
parser.add_argument('--target_kl', type=float, default=None, help='the target KL divergence threshold')
parser.add_argument('--num_minibatches', type=int, default=32, help='The number of mini-batches')
parser.add_argument('--minibatch_size', type=int, default=0, help='The mini-batch size (computed in runtime)')


def learn(args, agent, optimizer,training_data, device, writer):

    # Annealing the rate
    args.update_step = args.update_step + 1
    frac = 1.0 - (args.update_step - 1.0) / (args.num_iterations - 1.0)
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow

    # Training data
    states = torch.Tensor(training_data['state']).to(device)
    actions = torch.Tensor(training_data['action']).to(device)
    returns = training_data['return'].copy()
    returns = torch.Tensor(returns).to(device)
    logprobs = torch.Tensor(training_data['act_prob']).to(device)
    advantages = training_data['advantage'].copy()
    advantages = torch.Tensor(advantages).to(device)
    values = torch.Tensor(training_data['value']).to(device)

    # Advantage-normalization
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    b_inds = np.arange(args.batch_size)
    clipfracs = []
    # Updata for update_epochs
    for _ in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(states[mb_inds], actions[mb_inds])
            logratio = newlogprob - logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = advantages[mb_inds]
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
            # Entropy
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        
        if args.target_kl is not None and approx_kl > args.target_kl:
                break


    print("Loss:", loss.item())
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], args.update_step)
    writer.add_scalar("losses/loss", loss.item(), args.update_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), args.update_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), args.update_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), args.update_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), args.update_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), args.update_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), args.update_step)


def main():
    # Parse input parameters
    args, _ = parser.parse_known_args()

    args.pool_size = int(args.num_actors * 2048)
    args.batch_size = int(args.num_actors * 2048)
    args.training_freq = int(args.num_actors)
    args.num_iterations = int(args.num_steps // 2048)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    run_name = f"{args.env_id}__{args.alg}__{args.num_actors}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"LEARNER/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Expose socket to actors
    context = zmq.Context()
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:{args.param_port}')

    # Init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Make device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Creat envs and agent
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma) for i in range(args.num_envs)],
    )
    agent = Agent(envs).to(device)

    weights_socket.send(pickle.dumps(agent.state_dict()))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Variables to control the frequency of training
    receiving_condition = multiprocessing.Condition()
    num_receptions = multiprocessing.Value('i', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(capacity=args.pool_size)
    # Creating process
    Process(target=recv_data,
            args=(args.data_port, mem_pool, receiving_condition, num_receptions)).start()

    # Print throughput statistics
    Process(target=MultiprocessingMemPool.record_throughput, args=(mem_pool, args.record_throughput_interval)).start()

    for step in count(1):
        with receiving_condition:
            while num_receptions.value < args.training_freq:
                receiving_condition.wait()
            data = mem_pool.sample(size=args.batch_size)
            num_receptions.value -= args.training_freq
            # mem_pool.clear()
        # Training
        learn(args, agent, optimizer, data, device, writer)

        weights_socket.send(pickle.dumps(agent.state_dict()))


def recv_data(data_port, mem_pool, receiving_condition, num_receptions):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{data_port}')

    while True:
        # noinspection PyTypeChecker
        data = deserialize(data_socket.recv())
        data_socket.send(b'200')
        with receiving_condition:
            mem_pool.push(data)
            num_receptions.value += 1
            receiving_condition.notify()


if __name__ == '__main__':
    main()
