import argparse
import os
import random
import numpy as np
import torch as tc
from mpi4py import MPI
from cnn_policy import CnnPolicy
from common.atari_wrappers import make_atari, wrap_deepmind
import logger
from bench.monitor import Monitor
from pposgd_simple import learn
from play import play

p = argparse.ArgumentParser(description='Pytorch port of ppo1 for Atari.')
p.add_argument('--mode', choices=['train', 'play'], default='train')
p.add_argument('--agent_size', choices=['small', 'large'], default='large', help='Model size.')
p.add_argument('--max_timesteps', type=int, default=int(10 * 1e6), help='Total env steps.')
p.add_argument('--timesteps_per_actorbatch', type=int, default=256, help='Env steps per actor per improvement phase.')
p.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
p.add_argument('--gae_lambda', type=float, default=0.95, help='Decay param for GAE.')
p.add_argument('--clip_param', type=float, default=0.2, help='Clip param for PPO.')
p.add_argument('--ent_coef', type=float, default=0.01, help='Entropy bonus coefficient.')
p.add_argument('--optim_epochs', type=int, default=4, help='Epochs per policy improvement phase.')
p.add_argument('--optim_stepsize', type=float, default=3e-3, help='Adam stepsize parameter.')
p.add_argument('--optim_batchsize', type=int, default=64, help='State samples per gradient step per env copy.')
p.add_argument('--schedule', type=str, default='linear')
p.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Dir name for all checkpoints generated')
p.add_argument('--model_name', type=str, default='model-ppo1-defaults', help='Model name used for checkpoints')
p.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Environment name')
args = p.parse_args()

# get comm object and set separate torch seed per process since we sample actions using torch.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workerseed = 1868 + 10000 * comm.Get_rank()
tc.manual_seed(workerseed)
np.random.seed(workerseed)
random.seed(workerseed)

# configure logger.
if rank == 0:
    logger.configure()
else:
    logger.configure(format_strs=[])

# create env.
env = make_atari(args.env_name)
env.seed(workerseed)
env = Monitor(env, logger.get_dir() and
                    os.path.join(logger.get_dir(), str(rank)))
env = wrap_deepmind(env)
env.seed(workerseed)

# create agent.
agent = CnnPolicy(
    img_channels=env.observation_space.shape[-1],
    num_actions=env.action_space.n,
    kind=args.agent_size)

# optimizer and scheduler. grad steps is frac of env steps.
max_grad_steps = args.optim_epochs * args.max_timesteps // (comm.Get_size() * args.optim_batchsize)
optimizer = tc.optim.Adam(agent.parameters(), lr=args.optim_stepsize, eps=1e-5)
scheduler = tc.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, max_lr=args.optim_stepsize, total_steps=max_grad_steps,
    pct_start=0.0, anneal_strategy='linear', cycle_momentum=False,
    div_factor=1.0)

# currently we only support checkpointing for the model params and not the rest of it
# since we have not yet looked into synchronizing state for optimizers, schedulers etc. across processes.
if rank == 0:
    try:
        state_dict = tc.load(os.path.join(args.checkpoint_dir, args.model_name, 'model.pth'))
        agent.load_state_dict(state_dict)
        print(f"Continuing from checkpoint found at {os.path.join(args.checkpoint_dir, args.model_name, 'model.pth')}")
    except FileNotFoundError:
        print("Bad checkpoint or none on process 0. Continuing from scratch.")

# sync agent parameters from process with rank zero.
# analogous to
#
# https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py#L128
# https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam.py#L44
with tc.no_grad():
    for p in agent.parameters():
        p_data = p.data.numpy()
        comm.Bcast(p_data, root=0)
        p.data.copy_(tc.tensor(p_data).float())

if args.mode == 'train':
    learn(env=env, agent=agent, optimizer=optimizer, scheduler=scheduler, comm=comm,
          timesteps_per_actorbatch=args.timesteps_per_actorbatch,
          clip_param=args.clip_param, entcoeff=args.ent_coef,
          optim_epochs=args.optim_epochs, optim_batchsize=args.optim_batchsize,
          gamma=args.gamma, lam=args.gae_lambda, max_timesteps=args.max_timesteps,
          checkpoint_dir=args.checkpoint_dir, model_name=args.model_name)
    env.close()

elif args.mode == 'play':
    play(env=env, agent=agent, comm=comm, args=args)
    env.close()

else:
    raise NotImplementedError("Mode of operation not supported!")
