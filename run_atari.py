import argparse
import os
import random
import numpy as np
import torch as tc
from mpi4py import MPI
from common.atari_wrappers import make_atari, wrap_deepmind
import logger
from bench.monitor import Monitor
from cnn_policy import CnnPolicy
from pposgd_simple import learn
from play import play


def parse_args():
    p = argparse.ArgumentParser(description='Pytorch port of ppo1 for Atari.')
    p.add_argument('--mode', choices=['train', 'play'], default='train')
    p.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', help='Environment name')
    p.add_argument('--env_steps', type=int, default=int(40 * 1e6))
    p.add_argument('--timesteps_per_actorbatch', type=int, default=128)
    p.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    p.add_argument('--lam', type=float, default=0.95, help='Decay param for GAE')
    p.add_argument('--epsilon', type=float, default=0.1, help='Clip param for PPO')
    p.add_argument('--ent_coef', type=float, default=0.01, help='Entropy bonus coefficient')
    p.add_argument('--optim_epochs', type=int, default=3, help='Epochs per policy improvement phase')
    p.add_argument('--optim_stepsize', type=float, default=0.00025, help='Adam stepsize parameter')
    p.add_argument('--optim_batchsize', type=int, default=32, help='State samples per gradient step per actor')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Dir name for checkpoints generated')
    p.add_argument('--model_name', type=str, default='model-ppo-paper-defaults', help='Model name used for checkpoints')
    p.add_argument('--model_size', choices=['small', 'large'], default='large')
    p.add_argument('--seed', type=float, default=0)
    args = p.parse_args()
    return args


def main(args):
    # mpi communicator.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # seed.
    workerseed = args.seed + 10000 * comm.Get_rank() if args.seed is not None else None
    if workerseed is not None:
        tc.manual_seed(workerseed % 2 ** 32)
        np.random.seed(workerseed % 2 ** 32)
        random.seed(workerseed % 2 ** 32)

    # logger.
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    # env.
    env = make_atari(args.env_name)
    env.seed(workerseed)
    env = Monitor(env, logger.get_dir() and
              os.path.join(logger.get_dir(), str(rank)))

    env = wrap_deepmind(env, frame_stack=True)
    env.seed(workerseed)

    # agent.
    agent = CnnPolicy(
        img_channels=env.observation_space.shape[-1],
        num_actions=env.action_space.n,
        kind=args.model_size)

    # optimizer and scheduler.
    max_grad_steps = args.optim_epochs * args.env_steps // (comm.Get_size() * args.optim_batchsize)

    optimizer = tc.optim.Adam(agent.parameters(), lr=args.optim_stepsize, eps=1e-5)
    scheduler = tc.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.optim_stepsize, total_steps=max_grad_steps,
        pct_start=0.0, anneal_strategy='linear', cycle_momentum=False,
        div_factor=1.0)

    # checkpoint.
    if rank == 0:
        try:
            state_dict = tc.load(os.path.join(args.checkpoint_dir, args.model_name, 'model.pth'))
            agent.load_state_dict(state_dict)
            print(f"Continuing from checkpoint found at {os.path.join(args.checkpoint_dir, args.model_name, 'model.pth')}")
        except FileNotFoundError:
            print("Bad checkpoint or none on process 0. Continuing from scratch.")

    # sync.
    with tc.no_grad():
        for p in agent.parameters():
            p_data = p.data.numpy()
            comm.Bcast(p_data, root=0)
            p.data.copy_(tc.tensor(p_data).float())

    # operations.
    if args.mode == 'train':
        learn(env=env, agent=agent, optimizer=optimizer, scheduler=scheduler, comm=comm,
              timesteps_per_actorbatch=args.timesteps_per_actorbatch, max_timesteps=args.env_steps,
              optim_epochs=args.optim_epochs, optim_batchsize=args.optim_batchsize,
              gamma=args.gamma, lam=args.lam, clip_param=args.epsilon, entcoeff=args.ent_coef,
              checkpoint_dir=args.checkpoint_dir, model_name=args.model_name)
        env.close()

    elif args.mode == 'play':
        play(env=env, agent=agent, comm=comm, args=args)
        env.close()

    else:
        raise NotImplementedError("Mode of operation not supported!")


if __name__ == '__main__':
    # get command line input.
    args = parse_args()

    # run main.
    main(args)
