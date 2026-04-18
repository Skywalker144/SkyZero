import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, game, num_blocks=10, num_channels=128):
        super().__init__()
        self.board_size = game.board_size
        input_channels = game.num_planes

        self.start_layer = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.board_size * self.board_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_layer(x)
        for block in self.trunk:
            x = block(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
