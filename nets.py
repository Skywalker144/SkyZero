import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.silu(out)
        return out


class GlobalPoolingBias(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x):
        # x shape: [B, C, H, W]
        g = F.silu(self.bn(x))
        g_avg = torch.mean(g, dim=(2, 3))  # [B, C]
        g_max = torch.amax(g, dim=(2, 3))  # [B, C]
        g = torch.cat([g_avg, g_max], dim=1)  # [B, 2C]
        bias = self.fc(g).unsqueeze(-1).unsqueeze(-1)
        return bias


class GlobalPoolingResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.g_pool = GlobalPoolingBias(channels, channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)

        g_bias = self.g_pool(out)
        out = out + g_bias

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.silu(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_channels, board_size, head_channels=64):
        super().__init__()
        self.board_size = board_size

        self.conv_p = nn.Conv2d(in_channels, head_channels, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels, head_channels, kernel_size=1)

        self.g_pool = GlobalPoolingBias(head_channels, head_channels)
        self.bn = nn.BatchNorm2d(head_channels)

        self.conv_final = nn.Conv2d(head_channels, out_channels, kernel_size=1)

    def forward(self, x):
        p = self.conv_p(x)  # [B, in_channels, H, W] -> [B, head_channels, H, W]
        g = self.conv_g(x)  # [B, in_channels, H, W] -> [B, head_channels, H, W]

        p = p + self.g_pool(g)
        p = F.silu(self.bn(p))

        board_logits = self.conv_final(p)  # [B, out_channels, H, W]

        return board_logits


class ValueHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, head_channels=32, value_channels=64):
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, head_channels, kernel_size=1)

        self.fc1 = nn.Linear(head_channels * 3, value_channels)
        self.fc_value = nn.Linear(value_channels, out_channels)

    def forward(self, x):
        v = self.conv_v(x)
        v = F.silu(v)

        v_pooled = torch.cat([
            torch.mean(v, dim=(2, 3)),
            torch.amax(v, dim=(2, 3)),
            torch.std(v, dim=(2, 3))
        ], dim=1)

        out = F.silu(self.fc1(v_pooled))
        value_logits = self.fc_value(out)  # [B, 3]

        return value_logits


class ResNet(nn.Module):
    def __init__(self, game, num_blocks=4, num_channels=128):
        super().__init__()
        self.board_size = game.board_size
        input_channels = game.num_planes

        self.start_layer = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(inplace=True),
        )

        self.trunk_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if (i + 2) % 3 == 0:
                self.trunk_blocks.append(GlobalPoolingResBlock(num_channels))
            else:
                self.trunk_blocks.append(ResidualBlock(num_channels))

        self.total_policy_head = PolicyHead(num_channels, 2, self.board_size, head_channels=num_channels // 2)
        self.soft_policy_head = PolicyHead(num_channels, 1, self.board_size, head_channels=num_channels // 2)
        self.ownership_head = PolicyHead(num_channels, 1, self.board_size, head_channels=num_channels // 2)
        self.value_head = ValueHead(num_channels, 3, head_channels=num_channels // 4, value_channels=num_channels // 2)

    def forward(self, x):
        # x shape: [B, input_channels, H, W]
        x = self.start_layer(x)  # [B, input_channels, H, W] -> [B, num_channels, H, W]
        for block in self.trunk_blocks:
            x = block(x)

        total_policy_logits = self.total_policy_head(x)  # [B, num_channels, H, W] -> [B, 2, H, W]
        soft_policy_logits = self.soft_policy_head(x)  # [B, 1, H, W]
        ownership = self.ownership_head(x)  # [B, 1, H, W]
        value_logits = self.value_head(x)  # [B, num_channels, H, W] -> [B, 3] (win, draw, lose)

        nn_output = {
            'policy_logits': total_policy_logits[:, 0:1, :, :],
            'opponent_policy_logits': total_policy_logits[:, 1:2, :, :],
            'soft_policy_logits': soft_policy_logits,
            'ownership': ownership,
            'value_logits': value_logits,
        }
        return nn_output
