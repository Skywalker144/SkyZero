import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_channels, out_channels, stride) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = F.leaky_relu(x)
        return x


class NoisyLinear(nn.Module):
    """
    待修改-for AlphaZero
    """
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon.clone().detach())
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon.clone().detach())
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class ResNet(nn.Module):
    def __init__(self, game, num_blocks=4, num_channels=128):
        super().__init__()
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.residual_layers = ResidualLayer(num_channels, num_channels, num_blocks)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * game.board_size ** 2, game.board_size ** 2),
            # NoisyLinear(2 * game.board_size ** 2, game.board_size ** 2),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(game.board_size ** 2, 256),
            # NoisyLinear(game.board_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # NoisyLinear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_layer(x)
        x = self.residual_layers(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    # def reset_noise(self):
    #     for module in self.modules():
    #         if isinstance(module, NoisyLinear):
    #             module.reset_noise()
