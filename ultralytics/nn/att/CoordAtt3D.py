import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    """基础卷积模块：卷积 + 批归一化 + ReLU激活"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        # Branch1: 1x1
        self.branch1x1 = BasicConv2d(in_channels, in_channels//4, kernel_size=1)
        
        # Branch2: 3x3
        self.branch5x5_1 = BasicConv2d(in_channels, in_channels//4, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(in_channels//4, in_channels//4, kernel_size=3, padding=1)
        
        # Branch3: 5x5
        self.branch3x3_1 = BasicConv2d(in_channels, in_channels//4, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(in_channels//4, in_channels//4, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(in_channels//4, in_channels//4, kernel_size=3, padding=1)
        
        # Branch4: 平均池化 + 1x1卷积
        self.branch_pool = BasicConv2d(in_channels, in_channels//4, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3_1(x)
        branch3x3dbl = self.branch3x3_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3_3(branch3x3dbl)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)





class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt3D(nn.Module):
    def __init__(self, inp, reduction=16):
        super(CoordAtt3D, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))  # 新增：通道维度池化
        
        mip = max(8, inp // reduction)
        
        # 修改：为三个分支创建共享的降维层
        self.conv1_hw = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv1_c = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        
        # self.bn1_hw = nn.BatchNorm2d(mip)
        # self.bn1_c = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # 修改：增加通道注意力分支
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_c = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 新增：通道注意力

        self.inception = Inception(inp)
        
    def forward(self, x):
        
        x = self.inception(x)

        identity = x
        
        n, c, h, w = x.size()
        
        # 原有的高度和宽度池化
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]
        
        # 新增：通道池化
        x_c = self.pool_c(x)  # [n, c, 1, 1]
        
        # 处理高度和宽度分支
        y_hw = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        y_hw = self.conv1_hw(y_hw)
        # y_hw = self.bn1_hw(y_hw)
        y_hw = self.act(y_hw)
        
        # 处理通道分支
        y_c = self.conv1_c(x_c)  # [n, mip, 1, 1]
        # y_c = self.bn1_c(y_c)
        y_c = self.act(y_c)
        
        # 分割高度和宽度特征
        x_h_split, x_w_split = torch.split(y_hw, [h, w], dim=2)
        x_w_split = x_w_split.permute(0, 1, 3, 2)  # 恢复原始维度
        
        # 生成注意力权重
        a_h = self.conv_h(x_h_split).sigmoid()  # [n, oup, h, 1]
        a_w = self.conv_w(x_w_split).sigmoid()  # [n, oup, 1, w]
        a_c = self.conv_c(y_c).sigmoid()  # [n, oup, 1, 1]
        
        # 结合三个注意力分支
        # 方法1：直接相乘
        out = identity * (a_h * a_w * a_c) ** (1/3) + identity
        
        return out
    

