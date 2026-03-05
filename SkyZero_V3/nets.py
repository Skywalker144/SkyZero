import torch
import torch.nn as nn


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
        layer_mean = torch.mean(x, dim=(2, 3))
        layer_max = torch.amax(x, dim=(2, 3))
        return torch.cat((layer_mean, layer_max), dim=1)


class GPoolBias(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gpool = KataGPool()
        self.linear = nn.Linear(2 * in_channels, out_channels, bias=False)

    def forward(self, x):
        g = self.gpool(x)
        bias = self.linear(g).unsqueeze(-1).unsqueeze(-1)
        return bias


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.normactconv1 = NormActConv(channels, channels, kernel_size=3)
        self.normactconv2 = NormActConv(channels, channels, kernel_size=3)

    def forward(self, x):
        out = self.normactconv1(x)
        out = self.normactconv2(out)
        return x + out


class GlobalPoolingResidualBlock(nn.Module):
    def __init__(self, channels, gpool_channels=None):
        super().__init__()
        if gpool_channels is None:
            gpool_channels = channels
        self.pre_bn = nn.BatchNorm2d(channels)
        self.pre_act = nn.SiLU(inplace=True)
        self.regular_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, gpool_channels, kernel_size=3, padding=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(gpool_channels)
        self.gpool_act = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.gpool_to_bias = nn.Linear(gpool_channels * 2, channels, bias=False)
        self.normactconv2 = NormActConv(channels, channels, kernel_size=3)

    def forward(self, x):
        out = self.pre_bn(x)
        out = self.pre_act(out)

        regular = self.regular_conv(out)
        gpool = self.gpool_conv(out)
        gpool = self.gpool_bn(gpool)
        gpool = self.gpool_act(gpool)

        bias = self.gpool_to_bias(self.gpool(gpool)).unsqueeze(-1).unsqueeze(-1)
        regular = regular + bias

        regular = self.normactconv2(regular)
        return x + regular


class NestedBottleneckResBlock(nn.Module):
    def __init__(self, channels, mid_channels, internal_length=2, use_gpool=False):
        super().__init__()
        self.normactconvp = NormActConv(channels, mid_channels, kernel_size=1)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            if use_gpool and i == 0:
                self.blockstack.append(GlobalPoolingResidualBlock(mid_channels))
            else:
                self.blockstack.append(ResBlock(mid_channels))
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
        self.g_bn = nn.BatchNorm2d(head_channels)
        self.g_act = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(head_channels * 2, head_channels, bias=False)
        self.p_bn = nn.BatchNorm2d(head_channels)
        self.p_act = nn.SiLU(inplace=True)
        self.conv_final = nn.Conv2d(head_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        p = self.conv_p(x)
        g = self.conv_g(x)
        g = self.g_bn(g)
        g = self.g_act(g)
        g = self.gpool(g)
        g = self.linear_g(g).unsqueeze(-1).unsqueeze(-1)
        p = p + g
        p = self.p_bn(p)
        p = self.p_act(p)
        return self.conv_final(p)

class ValueHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, head_channels=32, value_channels=64):
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(head_channels)
        self.v_act = nn.SiLU(inplace=True)
        
        self.gpool = KataGPool()
        
        self.fc1 = nn.Linear(head_channels * 2, value_channels, bias=True)
        self.act2 = nn.SiLU(inplace=True)
        
        self.fc_value = nn.Linear(value_channels, out_channels, bias=True)

    def forward(self, x):
        v = self.conv_v(x)
        v = self.v_bn(v)
        v = self.v_act(v)  # [B, 1, H, W]
        
        v_pooled = self.gpool(v)
        out = self.act2(self.fc1(v_pooled))
        
        value_logits = self.fc_value(out)
        return value_logits
    
    
class ResNet(nn.Module):
    def __init__(self, game, num_blocks=6, num_channels=128):
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
        self.trunk_tip_bn = nn.BatchNorm2d(num_channels)
        self.trunk_tip_act = nn.SiLU(inplace=True)

        self.total_policy_head = PolicyHead(num_channels, 4, self.board_size, head_channels=num_channels // 2)
        self.value_head = ValueHead(num_channels, 3, head_channels=num_channels // 4, value_channels=num_channels // 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x):
        # x shape: [B, input_channels, H, W]
        x = self.start_layer(x)  # [B, input_channels, H, W] -> [B, num_channels, H, W]
        for block in self.trunk_blocks:
            x = block(x)
        x = self.trunk_tip_bn(x)
        x = self.trunk_tip_act(x)

        total_policy_logits = self.total_policy_head(x)  # [B, 4, H, W]
        
        value_logits = self.value_head(x)

        nn_output = {
            "policy_logits": total_policy_logits[:, 0:1, :, :],
            "opponent_policy_logits": total_policy_logits[:, 1:2, :, :],
            "soft_policy_logits": total_policy_logits[:, 2:3, :, :],
            "soft_opponent_policy_logits": total_policy_logits[:, 3:4, :, :],
            "value_logits": value_logits,
        }
        return nn_output