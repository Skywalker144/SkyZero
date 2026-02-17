import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPoolingBias(nn.Module):
    """
    KataGo 的 Global Pooling Bias 结构 (论文 Figure 2, Appendix A.2)。

    对 G 通道做全局池化 (mean + max) 得到 2*c_g 维向量，
    经 FC 映射为 c_x 维的逐通道 bias，加到 X 上。

    结构:
        G -> BN -> Act -> GlobalPool(mean, max) -> FC -> bias [c_x]
        X ─────────────────────────────────────── (+) bias -> output
    """
    def __init__(self, c_x, c_g):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_g)
        self.fc = nn.Linear(c_g * 2, c_x)

    def forward(self, x, g):
        # g: [B, c_g, H, W]
        g = F.silu(self.bn(g))
        # 全局池化: mean + max -> [B, 2*c_g]
        g_mean = g.mean(dim=(2, 3))                       # [B, c_g]
        g_max = g.amax(dim=(2, 3))                         # [B, c_g]
        g_pooled = torch.cat([g_mean, g_max], dim=1)       # [B, 2*c_g]
        # FC -> 逐通道 bias
        bias = self.fc(g_pooled)                           # [B, c_x]
        # channelwise addition
        return x + bias.unsqueeze(-1).unsqueeze(-1)        # [B, c_x, H, W]


class ResidualBlock(nn.Module):
    """
    Pre-activation 残差块 (He et al. 2016, KataGo 论文 Appendix A.3)。

    结构:
        identity ──────────────────────────────── (+)
              └─ BN → Act → Conv3x3                │
                            → BN → Act → Conv3x3 ──┘
    """
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.silu(self.bn1(x))
        out = self.conv1(out)
        out = F.silu(self.bn2(out))
        out = self.conv2(out)
        return x + out


class NestedBottleneckResidualBlock(nn.Module):
    """
    KataGo 的 Nested Bottleneck Residual Block (论文 Appendix / Gumbel MuZero Appendix)。
    
    结构:
        Input (C)
          │
        BN/Act/1x1 (C -> C/2)
          │
          ├── ResidualBlock (C/2) [2x 3x3 convs]
          │
          ├── ResidualBlock (C/2) [2x 3x3 convs]
          │
          ... (n_blocks 个)
          │
        BN/Act/1x1 (C/2 -> C)
          │
        (+) ──────────────────────> Output
    """
    def __init__(self, channels, n_blocks=2):
        super().__init__()
        c_internal = channels // 2
        
        # Entry 1x1 (Pre-activation: BN -> Act -> Conv)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, c_internal, kernel_size=1, bias=False)
        
        # Inner Blocks (Stack of standard ResidualBlocks)
        # KataGo 默认使用 2 个内部残差块 (共4个3x3卷积)
        self.inner_blocks = nn.Sequential(*[
            ResidualBlock(c_internal) for _ in range(n_blocks)
        ])
        
        # Exit 1x1 (Pre-activation: BN -> Act -> Conv)
        self.bn_out = nn.BatchNorm2d(c_internal)
        self.conv_out = nn.Conv2d(c_internal, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Entry
        out = F.silu(self.bn1(x))
        out = self.conv1(out)
        
        # Inner Stack
        out = self.inner_blocks(out)
        
        # Exit
        out = F.silu(self.bn_out(out))
        out = self.conv_out(out)
        
        return x + out


class GlobalPoolingResidualBlock(nn.Module):
    """
    KataGo 风格的全局池化 pre-activation 残差块 (论文 Appendix A.3)。

    与普通 ResidualBlock 的区别：在第一个 Conv3x3 之后，将前 c_pool 个
    通道通过 GlobalPoolingBias 结构映射为逐通道 bias，加到其余通道上。

    结构:
        identity ──────────────────────────────────────── (+)
                  └─ BN → Act → Conv3x3 ─┬─ G[:c_pool]    │
                                         ├─ X[c_pool:] ──(+bias)
                                         │   ↑ GlobalPoolingBias
                                         └─ BN → Act → Conv3x3 ─┘
    """
    def __init__(self, channels, c_pool=None):
        super().__init__()
        if c_pool is None:
            c_pool = max(32, channels // 4)
        self.c_pool = c_pool

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 全局池化 bias: 前 c_pool 通道 -> bias 到后 (channels - c_pool) 通道
        self.pool_bias = GlobalPoolingBias(c_x=channels - c_pool, c_g=c_pool)
        self.bn2 = nn.BatchNorm2d(channels - c_pool)
        self.conv2 = nn.Conv2d(channels - c_pool, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.silu(self.bn1(x))
        out = self.conv1(out)
        # 分割: 前 c_pool 通道作为全局池化来源 G，其余作为 X
        g = out[:, :self.c_pool]
        main = out[:, self.c_pool:]
        # 全局池化 bias: 加法
        main = self.pool_bias(main, g)
        # 第二层
        main = F.silu(self.bn2(main))
        out = self.conv2(main)
        return x + out


class ResNet(nn.Module):
    def __init__(self, game, num_blocks=4, num_channels=128, use_global_pool=True, use_nested_bottleneck=True, use_aux_policy=True):
        super().__init__()
        self.use_global_pool = use_global_pool
        self.use_aux_policy = use_aux_policy
        self.board_height = game.board_height
        self.board_width = game.board_width
        self.action_space_size = game.action_space_size

        input_channels = game.num_planes
        c = num_channels

        # 1. 输入层: 5x5 卷积 (论文 Appendix A.3)
        self.start_conv = nn.Conv2d(input_channels, c, kernel_size=5, stride=1, padding=2, bias=False)
        self.start_bn = nn.BatchNorm2d(c)

        # 2. 残差塔 (Backbone)
        # 每隔几个普通残差块插入一个全局池化残差块
        c_pool = max(32, c // 4)
        blocks = []
        for i in range(num_blocks):
            if use_global_pool and (i + 1) % 3 == 0:
                blocks.append(GlobalPoolingResidualBlock(c, c_pool))
            else:
                if use_nested_bottleneck:
                    blocks.append(NestedBottleneckResidualBlock(c))
                else:
                    blocks.append(ResidualBlock(c))
        self.backbone = nn.ModuleList(blocks)

        # 3. Trunk 末尾 BN + Act (pre-activation 架构需要)
        self.trunk_bn = nn.BatchNorm2d(c)

        # 4. Policy Head (论文 Appendix A.4)
        self.policy_head = self._build_policy_head(c)
        
        if self.use_aux_policy:
            self.aux_policy_head = self._build_policy_head(c)

        # Optimistic Policy Head (新增)
        self.optimistic_policy_head = self._build_policy_head(c)

        # 5. Value Head (论文 Appendix A.5)
        # 1x1 卷积 -> 全局池化 -> FC -> FC -> tanh
        c_val_head = max(32, c // 4)
        c_val_hidden = max(64, c)
        
        self.value_head = self._build_value_head(c, c_val_head, c_val_hidden, output_dim=1)
        
        # Short-term Value Head (新增, output_dim=3)
        self.short_term_value_head = self._build_value_head(c, c_val_head, c_val_hidden, output_dim=3)

        self._init_weights()

    def _build_policy_head(self, c):
        # 提取 Policy Head 构建逻辑以便复用
        # 并行两路 1x1 卷积: P 和 G
        board_cells = self.board_height * self.board_width
        spatial_policy = (self.action_space_size == board_cells)
        c_head = max(32, c // 4)
        
        conv_p = nn.Conv2d(c, c_head, kernel_size=1, bias=False)
        conv_g = nn.Conv2d(c, c_head, kernel_size=1, bias=False)
        # G 通过全局池化 bias 加到 P 上
        pool_bias = GlobalPoolingBias(c_x=c_head, c_g=c_head)
        bn = nn.BatchNorm2d(c_head)
        
        conv_out = None
        fc_out = None

        if spatial_policy:
            # 纯空间动作: 1x1 卷积输出 1 通道 (棋盘大小无关)
            conv_out = nn.Conv2d(c_head, 1, kernel_size=1)
        else:
            # 非纯空间动作 (如 Connect4 只选列): 全局池化后 FC 输出
            fc_out = nn.Linear(c_head * 2, self.action_space_size)
            
        return nn.ModuleDict({
            'conv_p': conv_p,
            'conv_g': conv_g,
            'pool_bias': pool_bias,
            'bn': bn,
            'conv_out': conv_out,
            'fc_out': fc_out
        })

    def _build_value_head(self, c, c_val_head, c_val_hidden, output_dim=1):
        return nn.ModuleDict({
            'conv': nn.Conv2d(c, c_val_head, kernel_size=1, bias=False),
            'bn': nn.BatchNorm2d(c_val_head),
            'fc1': nn.Linear(c_val_head * 2, c_val_hidden),
            'fc2': nn.Linear(c_val_hidden, output_dim)
        })

    def _forward_value_head(self, x, head_modules):
        v = head_modules['conv'](x)
        v = F.silu(head_modules['bn'](v))
        v_mean = v.mean(dim=(2, 3))
        v_max = v.amax(dim=(2, 3))
        v_pooled = torch.cat([v_mean, v_max], dim=1)
        v = F.silu(head_modules['fc1'](v_pooled))
        value = torch.tanh(head_modules['fc2'](v))
        return value

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward_policy_head(self, x, head_modules):
        p = head_modules['conv_p'](x)
        g = head_modules['conv_g'](x)
        p = head_modules['pool_bias'](p, g)
        p = F.silu(head_modules['bn'](p))
        
        if head_modules['conv_out'] is not None:
            # Spatial Policy
            policy = head_modules['conv_out'](p)
            policy = policy.flatten(1)
        else:
            # Non-spatial Policy
            p_mean = p.mean(dim=(2, 3))
            p_max = p.amax(dim=(2, 3))
            p_pooled = torch.cat([p_mean, p_max], dim=1)
            policy = head_modules['fc_out'](p_pooled)
            
        return policy

    def forward(self, x):
        # x shape: [B, input_channels, H, W]

        # 输入层: 5x5 conv (pre-activation 架构下，起始层自带 BN+Act)
        x = F.silu(self.start_bn(self.start_conv(x)))

        # 残差塔
        for block in self.backbone:
            x = block(x)

        # Trunk 末尾激活 (pre-activation 架构: 最后一个残差块的输出未经 BN+Act)
        x = F.silu(self.trunk_bn(x))

        # Policy Head
        policy = self.forward_policy_head(x, self.policy_head)
        
        # Auxiliary Policy Head
        aux_policy = None
        if self.use_aux_policy and self.training:
            aux_policy = self.forward_policy_head(x, self.aux_policy_head)

        # Optimistic Policy Head
        optimistic_policy = None
        if self.training:
            optimistic_policy = self.forward_policy_head(x, self.optimistic_policy_head)

        # Value Head
        value = self._forward_value_head(x, self.value_head)

        # Short-term Value Head
        short_term_value = None
        if self.training:
            short_term_value = self._forward_value_head(x, self.short_term_value_head)

        if self.training:
            return policy, value, aux_policy, optimistic_policy, short_term_value
            
        return policy, value

