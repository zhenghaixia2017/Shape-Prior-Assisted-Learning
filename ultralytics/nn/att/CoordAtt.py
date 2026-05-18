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


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=16):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.inception = Inception(inp)

    def forward(self, x):
        x = self.inception(x)
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        att = (a_w * a_h) ** (1/2)

        out = identity * att  + identity

        return out
