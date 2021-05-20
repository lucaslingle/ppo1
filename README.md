# ppo1
Pytorch port of [OpenAI baselines ppo1](https://github.com/openai/baselines/tree/master/baselines/ppo1), supporting parallel experience collection. 

### Background
Proximal Policy Optimization is a reinforcement learning algorithm proposed by [Schulman et al., 2017](https://arxiv.org/abs/1707.06347). 
Compared to vanilla policy gradients and/or actor-critic methods, which optimize the model parameters by estimating the gradient of the reward surface
and taking a single step, PPO takes inspiration from an approximate natural policy gradient algorithm known as TRPO.

[TRPO](https://arxiv.org/abs/1502.05477) is an example of an information-geometric trust region method, which aims to improve the policy by taking steps of a constant maximum size on the manifold of possible policies.
The metric utilized in TRPO is the state-averaged KL divergence under the current policy; taking steps under TRPO amounts to solving a constrained optimization problem 
to ensure the step size is at most a certain amount. This is done using conjugate gradient ascent to compute the (approximate) natural gradient, followed by a line search 
to ensure the step taken in parameter space leads to a policy whose state-averaged KL divergence to the previous policy is not larger than a certain amount. 

Compared to vanilla policy gradients and/or actor-critic methods, the PPO algorithm enjoys the following favorable properties:
- Improved sample efficiency
- Improved stability
- Improved reusability of collected experience

Compared to TRPO, proximal policy optimization is considerably simpler, easier to implement, and allows recurrent policies without any additional complication. 
This repo implements the current commit of OpenAI baselines' ppo1 (commit 8a97e0d); ppo1 was originally released as the reference implementation for Schulman et al., 2017. 

There is also a ppo2, but it performs poorly compared to ppo1 on games like Pong, and the reason for its release was, to our knowledge, never provided. 
Furthermore, it relies on a clipping heuristic to construct a pessimistic value function, and the efficacy of the value clipping heuristic depends on the scale of the rewards.
Although it works for environments found in the Atari 2600 suite, it requires a reward wrapper that clips the rewards to [-1, 1], which could be unacceptable in some domains.
It also requires gradient clipping with a maximum gradient norm of 0.5, which is only acceptable for some architectures.

Thus, ppo1 is in some respects a much more general algorithm. 

### Getting Started

Install the following system dependencies:
##### Ubuntu     
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

##### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

##### Everyone
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

Then run
``` 
conda create --name ppo1 python=3.8.1
conda activate ppo1
git clone https://github.com/lucaslingle/ppo1
cd ppo1
conda install --file requirements.txt
```


