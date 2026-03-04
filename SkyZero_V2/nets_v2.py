import torch
import torch.nn as nn


class Bias2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.beta


class NormActConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU(inplace=True)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return self.conv(x)


class KataGPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [B, C, H, W]
        h = x.shape[2]
        w = x.shape[3]
        area = float(h * w)
        area_sqrt_offset = (area ** 0.5) - 14.0

        layer_mean = torch.mean(x, dim=(2, 3))
        layer_max = torch.amax(x, dim=(2, 3))
        out_pool1 = layer_mean
        out_pool2 = layer_mean * (area_sqrt_offset / 10.0)
        out_pool3 = layer_max
        return torch.cat((out_pool1, out_pool2, out_pool3), dim=1)


class GPoolBias(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gpool = KataGPool()
        self.linear = nn.Linear(3 * in_channels, out_channels, bias=False)

    def forward(self, x):
        g = self.gpool(x)
        bias = self.linear(g).unsqueeze(-1).unsqueeze(-1)
        return bias


class ResBlock(nn.Module):
    def __init__(self, channels, use_gpool=False):
        super().__init__()
        self.use_gpool = use_gpool
        self.normactconv1 = NormActConv(channels, channels, kernel_size=3)
        self.normactconv2 = NormActConv(channels, channels, kernel_size=3)
        self.gpool_bias = GPoolBias(channels, channels) if use_gpool else None

    def forward(self, x):
        out = self.normactconv1(x)
        if self.gpool_bias is not None:
            out = out + self.gpool_bias(out)
        out = self.normactconv2(out)
        return x + out


class NestedBottleneckResBlock(nn.Module):
    def __init__(self, channels, mid_channels, internal_length=2, use_gpool=False):
        super().__init__()
        self.normactconvp = NormActConv(channels, mid_channels, kernel_size=1)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            self.blockstack.append(ResBlock(mid_channels, use_gpool=(use_gpool and i == 0)))
        self.normactconvq = NormActConv(mid_channels, channels, kernel_size=1)

    def forward(self, x):
        out = self.normactconvp(x)
        for block in self.blockstack:
            out = block(out)
        out = self.normactconvq(out)
        return x + out


class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_channels, board_size, head_channels=64):
        super().__init__()
        self.board_size = board_size

        self.conv_p = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.conv_g = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * head_channels, head_channels, bias=False)
        self.bias = Bias2d(head_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_final = nn.Conv2d(head_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        p = self.conv_p(x)
        g = self.conv_g(x)
        g = self.gpool(g)
        g = self.linear_g(g).unsqueeze(-1).unsqueeze(-1)
        p = p + g
        p = self.bias(p)
        p = self.act(p)
        return self.conv_final(p)


class ValueHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, head_channels=32, value_channels=64):
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.bias = Bias2d(head_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.fc1 = nn.Linear(head_channels * 3, value_channels, bias=True)
        self.act2 = nn.SiLU(inplace=True)
        self.fc_value = nn.Linear(value_channels, out_channels, bias=True)
        self.conv_ownership = nn.Conv2d(head_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        v = self.conv_v(x)
        v = self.bias(v)
        v = self.act1(v)

        v_pooled = self.gpool(v)
        out = self.act2(self.fc1(v_pooled))
        value_logits = self.fc_value(out)
        ownership = self.conv_ownership(v)

        return value_logits, ownership


class ResNet(nn.Module):
    def __init__(self, game, num_blocks=4, num_channels=128):
        super().__init__()
        self.board_size = game.board_size
        input_channels = game.num_planes
        mid_channels = max(16, num_channels // 2)

        self.start_layer = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(inplace=True),
        )

        self.trunk_blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = (i + 2) % 3 == 0
            self.trunk_blocks.append(
                NestedBottleneckResBlock(
                    channels=num_channels,
                    mid_channels=mid_channels,
                    internal_length=2,
                    use_gpool=use_gpool,
                )
            )

        self.total_policy_head = PolicyHead(num_channels, 2, self.board_size, head_channels=num_channels // 2)
        self.soft_policy_head = PolicyHead(num_channels, 1, self.board_size, head_channels=num_channels // 2)
        self.value_head = ValueHead(num_channels, 3, head_channels=num_channels // 4, value_channels=num_channels // 2)

    def forward(self, x):
        # x shape: [B, input_channels, H, W]
        x = self.start_layer(x)  # [B, input_channels, H, W] -> [B, num_channels, H, W]
        for block in self.trunk_blocks:
            x = block(x)

        total_policy_logits = self.total_policy_head(x)  # [B, num_channels, H, W] -> [B, 2, H, W]
        soft_policy_logits = self.soft_policy_head(x)  # [B, 1, H, W]
        value_logits, ownership = self.value_head(x)

        nn_output = {
            "policy_logits": total_policy_logits[:, 0:1, :, :],
            "opponent_policy_logits": total_policy_logits[:, 1:2, :, :],
            "soft_policy_logits": soft_policy_logits,
            "ownership": ownership,
            "value_logits": value_logits,
        }
        return nn_output
