# ppo1
Pytorch port of [OpenAI baselines ppo1](https://github.com/openai/baselines/tree/master/baselines/ppo1), supporting parallel experience collection. 

![pong gif](assets/model-ppo1-defaults/pong.gif)
![breakout gif](assets/breakout-ppo-paper-defaults/breakout.gif)
![mspacman gif](assets/mspacman-ppo-paper-defaults/mspacman.gif)

## Background
Proximal Policy Optimization is a reinforcement learning algorithm proposed by [Schulman et al., 2017](https://arxiv.org/abs/1707.06347). 
Compared to vanilla policy gradients and/or actor-critic methods, which optimize the model parameters by estimating the gradient of the reward surface
and taking a single step, PPO takes inspiration from an approximate natural policy gradient algorithm known as TRPO.

[TRPO](https://arxiv.org/abs/1502.05477) is an example of an information-geometric trust region method, which aims to improve the policy by taking steps of a constant maximum size on the manifold of possible policies.
The metric utilized in TRPO is the state-averaged KL divergence under the current policy; taking steps under TRPO amounts to solving a constrained optimization problem 
to ensure the step size is at most a certain amount. This is done using conjugate gradient descent to compute the (approximate) natural gradient, followed by a line search 
to ensure the step taken in parameter space leads to a policy whose state-averaged KL divergence to the previous policy is not larger than a certain amount. 

Compared to vanilla policy gradients and/or actor-critic methods, the PPO algorithm enjoys the following favorable properties:
- Improved sample efficiency
- Improved stability
- Improved reusability of collected experience

Compared to TRPO, proximal policy optimization is considerably simpler, easier to implement, and allows recurrent policies without any additional complication. 
This repo implements the current commit of OpenAI baselines' ppo1 (commit 8a97e0d); ppo1 was originally released as the reference implementation for Schulman et al., 2017. 

There is also a ppo2, but it performs poorly compared to ppo1 on games like Pong, and the reason for its release was, to our knowledge, never provided. 
Furthermore, it relies on a clipping heuristic to construct a pessimistic value function loss, and the efficacy of the value clipping heuristic depends on the scale of the rewards.
Although it works for environments found in the Atari 2600 suite, it requires a reward wrapper that clips the rewards to [-1, 1], which could be unacceptable in some domains.
It also requires gradient clipping with a maximum gradient norm of 0.5, which is only acceptable for some architectures.

Thus, ppo1 is in some respects a much more general algorithm. 

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Everyone
Once the system dependencies have been installed, it's time to install the python dependencies. 
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

Then run
```bash
git clone https://github.com/lucaslingle/ppo1
cd ppo1
conda env create -f environment.yml python=3.8.1
conda activate ppo1
```

## Usage

#### Training
To run the default settings, you can simply type:
```bash
mpirun -np 8 python -m run_atari --env_name=PongNoFrameskip-v4
```

This will launch 8 parallel processes, each running the ```run_atari.py``` script. These processes will play the OpenAI gym environment 'PongNoFrameskip-v4' in parallel, 
and communicate gradient information and synchronize parameters using [OpenMPI](https://www.open-mpi.org/).

To see additional options, you can simply type ```python run_atari.py --help```. In particular, you can pick any other Atari 2600 game supported by OpenAI gym, 
and this implementation will support it. 

#### Checkpoints
By default, checkpoints are saved to ```./checkpoints/model-ppo1-defaults```. To pick a different checkpoint directory, 
you can set the ```--checkpoint_dir``` flag, and to pick a different checkpoint name, you can set the ```--model_name``` flag.

#### Play
To watch the trained agent play a game, you can run
```bash
mpirun -np 8 python -m run_atari --env_name=PongNoFrameskip-v4 --mode=play
```
Be sure to use the correct env_name, and to pass in the appropriate checkpoint_dir and model_name.

## Acknowledgements

Over 90% of the code in this repo comes from OpenAI baselines. That said, in porting their Tensorflow implementation of ppo1 to Pytorch, we have made the following changes:

* Standard Adam optimizer. The OpenAI baselines repo uses a custom MpiAdam class to implement inter-process gradient computation and to sync model parameters at the start of training.
  It also implements Adam slightly differently than the original authors proposed, resulting an Adam epsilon that is essentially rescaled over time by ```np.sqrt(1-beta2**t)```. We did not find this modification to improve performance, so we use the default Adam optimizer in Pytorch, and compute inter-process gradients before passing them to the Adam optimizer on each process.
* Torch distributions. We define an action distribution using Pytorch's built-in torch.distributions.Categorical class, rather than using custom classes as done in OpenAI baselines. As a result, our models sample from the categorical distribution using a softmax on the logits, followed by a uniform search (the Pytorch default), rather than sampling in log-space using the gumbel-max trick. We also compute entropy and log-probabilities using the built-in methods in the torch.distributions.Categorical class.
* Normc initialization. Tensorflow and Pytorch store weight tensors for linear layers differently, and thus a faithful implementation of the ```normc_initializer``` from Baselines requires normalizing over a different axis to obtain the same initialization. 
* Single network only. We store only one set of model parameters, rather than two. Instead of maintaining an old network to compute the demoninators for the policy ratio, we save the relevant quantities during experience collection. This makes the implementation more lightweight without affecting the underlying computation.
* No kl divergence logging. We do not log the approximate KL divergence between the old policy and the new one, since this would require storing the entire vector of policy probabilities during experience collection. However, support for this can be easily added.
* Framestack. We add a frame stack of size 4 to the wrapped environments. This frame stacking operation is distinct from frame skipping (see [here](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)), and is useful in some games to infer the direction of moving objects. It is currently absent from the latest commit of ppo1, although frame stacking [is standard](https://github.com/openai/baselines/blob/master/baselines/run.py#L103) in the other baselines algorithms, and was originally present in the [initial commit](https://github.com/openai/baselines/commit/d9f194f797f406969f454ba7338c798c14cff01e#diff-c9410d962ac09d675492e6638b87de62271d27cf85ef07e584a861e27d633b98) of ppo1. Frame stacking is required to obtain good performance on games like Breakout. 
* Clip parameter annealing. This implementation follows the Atari hyperparameters provided by Schulman et al., 2017 for Atari, and thus includes the linear annealing of the PPO clip parameters to zero over the course of training. This was initially implemented in ppo1, but was later removed without an accompanying explanation in the commit message. 
* Standard value function loss. We use the standard mean-squared error loss function, following Schulman et al., 2017. In the initial public commit of ppo1, the pessimistic value loss was used, but this was later moved to ppo2, and does not appear in their paper, and does not appear to be not necessary to reproduce their results. 
* Video Recording. Original implementation for saving video footage of the trained agent.

Huge thanks to OpenAI for maintaining baselines and releasing the ppo1 implementation!

