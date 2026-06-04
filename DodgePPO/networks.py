"""Actor-Critic network for PPO.

A single :class:`ActorCritic` covers the dodge env's two observation modes and
two action modes:

* **vector** obs ``(D,)``  -> MLP torso
* **grid**   obs ``(C,H,W)`` -> small CNN torso
* **discrete** actions     -> Categorical policy head (logits)
* **continuous** actions   -> diagonal Gaussian head (state-independent log-std)

Orthogonal initialization with the usual PPO gains (sqrt(2) for hidden layers,
0.01 for the policy head so the initial policy is near-uniform, 1.0 for the value
head).  ``get_action_and_value`` returns ``(action, log_prob, entropy, value)`` —
the standard CleanRL-style interface the training loop consumes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _MLPTorso(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x):
        return self.net(x)


class _CNNTorso(nn.Module):
    """Two 2x2 conv layers + an MLP head (matches the dodge grid encoding)."""

    def __init__(self, obs_shape, channels=(32, 64), hidden=(256,)):
        super().__init__()
        c, h, w = obs_shape
        conv, last = [], c
        for ch in channels:
            conv += [layer_init(nn.Conv2d(last, ch, kernel_size=2)), nn.ReLU()]
            last = ch
        self.conv = nn.Sequential(*conv)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        fc, last = [], n
        for hd in hidden:
            fc += [layer_init(nn.Linear(last, hd)), nn.ReLU()]
            last = hd
        self.fc = nn.Sequential(*fc)
        self.out_dim = last

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.fc(self.conv(x).flatten(1))


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, *, num_actions=None, act_dim=None,
                 continuous=False, hidden=(256, 256), channels=(32, 64)):
        super().__init__()
        self.continuous = bool(continuous)
        obs_shape = tuple(obs_shape)
        if len(obs_shape) == 1:
            self.torso = _MLPTorso(obs_shape[0], hidden)
        else:
            self.torso = _CNNTorso(obs_shape, channels, hidden[:1] or (256,))
        feat = self.torso.out_dim

        self.critic = layer_init(nn.Linear(feat, 1), std=1.0)
        if self.continuous:
            assert act_dim is not None
            self.act_dim = int(act_dim)
            self.actor_mean = layer_init(nn.Linear(feat, self.act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))
        else:
            assert num_actions is not None
            self.num_actions = int(num_actions)
            self.actor = layer_init(nn.Linear(feat, self.num_actions), std=0.01)

    def get_value(self, obs):
        return self.critic(self.torso(obs)).squeeze(-1)

    def _dist(self, feat):
        if self.continuous:
            mean = self.actor_mean(feat)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            return Normal(mean, std)
        return Categorical(logits=self.actor(feat))

    def get_action_and_value(self, obs, action=None):
        feat = self.torso(obs)
        dist = self._dist(feat)
        if action is None:
            action = dist.sample()
        value = self.critic(feat).squeeze(-1)
        if self.continuous:
            # sum log-probs / entropy across the action dimensions
            logp = dist.log_prob(action).sum(-1)
            ent = dist.entropy().sum(-1)
        else:
            logp = dist.log_prob(action)
            ent = dist.entropy()
        return action, logp, ent, value

    @torch.no_grad()
    def act_greedy(self, obs):
        """Deterministic action for evaluation (argmax / distribution mean)."""
        feat = self.torso(obs)
        if self.continuous:
            return self.actor_mean(feat)
        return self.actor(feat).argmax(-1)
