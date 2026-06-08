"""SAC networks for the *continuous* Channel-Dodge action mode (``Box(-1,1,(2,))``).

* :class:`SquashedGaussianActor` — an MLP that outputs a diagonal Gaussian over the
  pre-squash action, then ``tanh`` to bound it to ``(-1, 1)``; ``sample`` returns
  the action and its ``tanh``-corrected log-prob. ``mean_action`` gives the
  deterministic ``tanh(mean)`` for evaluation.
* :class:`TwinQ` — two independent Q(s,a) critics (the clipped-double-Q trick);
  both are returned so the trainer can take their min.

Plain ``Linear`` + ReLU MLPs (no NoisyNet here — SAC explores via its stochastic
policy and the entropy term).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _mlp(in_dim, hidden, out_dim):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.act_dim = int(act_dim)
        self.net = _mlp(obs_dim, hidden, 2 * self.act_dim)

    def _mean_logstd(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self._mean_logstd(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()                       # reparameterized
        a = torch.tanh(x)
        # tanh change-of-variables correction for the log-prob
        logp = normal.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        logp = logp.sum(-1)
        return a, logp

    @torch.no_grad()
    def mean_action(self, obs):
        mu, _ = self._mean_logstd(obs)
        return torch.tanh(mu)


class DeterministicActor(nn.Module):
    """TD3 actor: a plain MLP squashed by tanh to (-1, 1) — deterministic policy."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.act_dim = int(act_dim)
        self.net = _mlp(obs_dim, hidden, self.act_dim)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
