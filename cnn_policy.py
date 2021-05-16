"""
Pytorch port of the cnn_policy.py from https://github.com/openai/baselines/ppo1/
"""

import torch as tc
from torch.distributions import Categorical
import numpy as np


def normc_init(weight_tensor, gain=1.0):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97

    Note that in tensorflow the weight tensor in a linear layer is stored with the
    input dim first and the output dim second. See
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/layers/core.py#L1193

    In contrast, in pytorch the output dim is first. See
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

    This means if we want a normc init in pytorch,
    we have to change which dim(s) we normalize on.

    We currently only support normc init for linear layers.
    Performance not guaranteed with other layer weight types.
    """
    with tc.no_grad():
        out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.size())
        out = gain * out / np.sqrt(np.sum(np.square(out), axis=1, keepdims=True))
        weight_tensor.copy_(tc.tensor(out))


class CnnPolicy(tc.nn.Module):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py

    This ported implementation only supports discrete-action games with 84x84 inputs.
    Since the CnnPolicy is only used for Atari games, this suffices.
    """
    def __init__(self, img_channels, num_actions, kind='large'):
        assert kind in ['small', 'large']
        super().__init__()
        self.num_actions = num_actions
        self.feature_dim = 512 if kind == 'large' else 256
        if kind == 'small':
            self.conv_stack = tc.nn.Sequential(
                tc.nn.Conv2d(img_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
                tc.nn.ReLU(),
                tc.nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
                tc.nn.ReLU(),
                tc.nn.Flatten(),
                tc.nn.Linear(2592, self.feature_dim),
                tc.nn.ReLU()
            )
        elif kind == 'large':
            self.conv_stack = tc.nn.Sequential(
                tc.nn.Conv2d(img_channels, 32, kernel_size=(8, 8), stride=(4, 4)),
                tc.nn.ReLU(),
                tc.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                tc.nn.ReLU(),
                tc.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                tc.nn.ReLU(),
                tc.nn.Flatten(),
                tc.nn.Linear(3136, self.feature_dim),
                tc.nn.ReLU(),
            )
        else:
            raise NotImplementedError

        for m in self.conv_stack.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.xavier_uniform_(m.weight)
                tc.nn.init.zeros_(m.bias)
            elif isinstance(m, tc.nn.Linear):
                normc_init(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

        self.policy_head = PolicyHead(self.feature_dim, self.num_actions)
        self.value_head = ValueHead(self.feature_dim)

    def forward(self, x):
        assert x.shape[1] == x.shape[2] == 84
        x = x / 255.
        x = x.permute(0, 3, 1, 2)
        x = x.detach()
        features = self.conv_stack(x)
        dist_pi = self.policy_head(features)
        vpred = self.value_head(features)
        return dist_pi, vpred


class PolicyHead(tc.nn.Module):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py#L36
    """
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.policy_head = tc.nn.Sequential(
            tc.nn.Linear(self.num_features, self.num_actions),
        )
        for m in self.policy_head.modules():
            if isinstance(m, tc.nn.Linear):
                normc_init(m.weight, gain=0.01)
                tc.nn.init.zeros_(m.bias)

    def forward(self, features):
        logits = self.policy_head(features)
        dist = Categorical(logits=logits)
        return dist


class ValueHead(tc.nn.Module):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py#L38
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.value_head = tc.nn.Sequential(
            tc.nn.Linear(self.num_features, 1)
        )

        for m in self.value_head.modules():
            if isinstance(m, tc.nn.Linear):
                normc_init(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

    def forward(self, features):
        return self.value_head(features)
