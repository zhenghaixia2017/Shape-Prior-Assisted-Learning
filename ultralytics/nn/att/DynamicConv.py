import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # # 静态基础卷积（提供稳定性）
        # self.static_conv = nn.Conv2d(
        #     in_channels, in_channels, kernel_size,
        #     padding=kernel_size//2, groups=in_channels, bias=False
        # )
        
        # 参数生成网络（生成残差权重）
        self.condition_net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 
                in_channels * kernel_size * kernel_size
            )
        )
        
        # 归一化层
        self.norm = nn.BatchNorm2d(in_channels)
        # self.norm_point = nn.BatchNorm2d(in_channels)
        
        # 缩放因子（控制动态权重强度）
        # self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始较小
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        
        # 静态卷积路径
        # static_out = self.static_conv(x)
        
        # ========== 动态路径 ==========
        # 生成动态参数
        x_gap = self.gap(x).view(B, -1)
        params = self.condition_net(x_gap)
        
        # 分割参数并归一化
        depth_param_size = C * k * k
        point_param_size = C * C
        
        depth_params = params
        
        # 归一化动态权重
        depth_params = torch.tanh(depth_params)
        # point_params = torch.tanh(point_params)
        
        # ========== 动态深度卷积 ==========
        depth_weights = depth_params.view(B, C, k*k)
        
        x_unfolded = F.unfold(x, kernel_size=k, padding=self.padding)
        L = x_unfolded.shape[-1]
        x_unfolded_reshaped = x_unfolded.view(B, C, k*k, L)
        
        # 动态卷积计算
        dynamic_depth = torch.einsum('bck,bckl->bcl', depth_weights, x_unfolded_reshaped)
        dynamic_depth = dynamic_depth.view(B, C, H, W)
        
        # ========== 融合静态和动态 ==========
        # depth_out = dynamic_depth
        # depth_out = self.norm_depth(depth_out)
        
        # ========== 动态逐点卷积 ==========
        # point_weights = point_params.view(B, C, C)
        # depth_out_flat = depth_out.view(B, C, -1)
        
        # dynamic_point = torch.einsum('boc,bcl->bol', point_weights, depth_out_flat)
        # dynamic_point = dynamic_point.view(B, C, H, W)
        
        # 最终输出 = 输入 + 动态残差
        output = x + dynamic_depth
        output = self.norm(output)
        
        return output