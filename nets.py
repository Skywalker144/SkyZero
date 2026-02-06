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

class GlobalPoolingBlock(nn.Module):
    """KataGo 风格的全局池化模块 - 用于获取全盘语境"""
    def __init__(self, in_channels, reduced_channels=64):
        super().__init__()
        # 全局池化后的特征降维
        self.pool_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 全局平均池化和最大池化
        mean_pool = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        max_pool = F.adaptive_max_pool2d(x, 1)   # [B, C, 1, 1]
        
        # 拼接
        global_features = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2C, 1, 1]
        
        # 降维
        global_features = self.pool_conv(global_features)  # [B, reduced, 1, 1]
        
        # 广播 - expand 不会复制内存，比 repeat 高效
        global_features = global_features.expand(-1, -1, height, width)
        
        # 拼接到原特征图上
        return torch.cat([x, global_features], dim=1)

class ResNet(nn.Module):
    def __init__(self, game, num_blocks=4, num_channels=128, use_global_pool=True):
        super().__init__()
        self.use_global_pool = use_global_pool
        self.board_height = game.board_height
        self.board_width = game.board_width
        self.action_space_size = game.action_space_size
        
        # 1. 输入处理层
        # 输入通道 = (历史步数 * 2 (黑白平面)) + 1 (当前颜色平面)
        input_channels = game.num_planes
        self.start_layer = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(inplace=True),
        )
        
        # 2. 残差塔 (Backbone)
        self.backbone = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        
        # 3. 全局池化
        if use_global_pool:
            g_channels = num_channels // 4
            self.global_pool = GlobalPoolingBlock(num_channels, reduced_channels=g_channels)
            head_in_channels = num_channels + g_channels
        else:
            head_in_channels = num_channels
        
        board_cells = self.board_height * self.board_width
        
        # 4. Policy Head
        # 通道数与网络宽度挂钩，例如设为 num_channels // 8 (至少为2)
        policy_head_channels = max(2, num_channels // 8)
        self.policy_head = nn.Sequential(
            nn.Conv2d(head_in_channels, policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_head_channels),
            nn.SiLU(),
            nn.Flatten(),
            # 注意：这里假设不包含 Pass 动作，如果包含，通常输出维度是 action_space_size + 1
            nn.Linear(policy_head_channels * board_cells, self.action_space_size),
        )

        # 5. Value Head
        # 隐藏层大小与网络宽度挂钩，例如设为 num_channels * 2
        value_hidden_size = num_channels * 2
        self.value_head = nn.Sequential(
            nn.Conv2d(head_in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(board_cells, value_hidden_size),
            nn.SiLU(),
            nn.Linear(value_hidden_size, 1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [B, input_channels, H, W]
        x = self.start_layer(x)
        x = self.backbone(x)
        
        if self.use_global_pool:
            x = self.global_pool(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
