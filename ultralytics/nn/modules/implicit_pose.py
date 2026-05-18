# ultralytics/nn/modules/implicit_pose.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import C2f   
import copy

class ImplicitShapeEncoder(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        # self.channel_attn = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(in_channels, in_channels // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_channels // reduction, in_channels),
        #     nn.Sigmoid()
        # )
        # self.project = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        spatial_weight = self.spatial_attn(x)
        spatial_out = x * spatial_weight
        # channel_weight = self.channel_attn(x).view(x.size(0), x.size(1), 1, 1)
        # channel_out = x * channel_weight
        # combined = spatial_out + channel_out
        return spatial_out


class ShapeAwareImplicitPose(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.target_branch = nn.Conv2d(in_channels, out_channels, 1) #C2f(in_channels, out_channels, n=1)
        self.context_branch = nn.Conv2d(in_channels, out_channels, 1) #C2f(in_channels, out_channels, n=1)
        self.shape_encoder = ImplicitShapeEncoder(out_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        target = self.target_branch(x)
        context = self.context_branch(x)
        shape = self.shape_encoder(target)        
        self.shape_feat = shape                     
        combined = torch.cat([context, shape], dim=1)
        return self.fusion(combined)
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'shape_feat':
                setattr(result, k, None)  
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


# class ShapeAwareImplicitPose(nn.Module):
#     def __init__(self, in_channels, out_channels=None):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         self.target_branch = nn.Conv2d(in_channels, out_channels, 1) #C2f(in_channels, out_channels, n=1)
#         self.context_branch = nn.Conv2d(in_channels, out_channels, 1) #C2f(in_channels, out_channels, n=1)
#         self.shape_encoder = ImplicitShapeEncoder(out_channels)
#         self.fusion = nn.Sequential(
#             nn.Conv2d(out_channels * 2, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # target = self.target_branch(x)
#         context = x # self.context_branch(x)
#         shape = self.shape_encoder(x)        
#         self.shape_feat = shape                     
#         combined = torch.cat([context, shape], dim=1)
#         return self.fusion(combined)
#     def __deepcopy__(self, memo):
#         cls = self.__class__
#         result = cls.__new__(cls)
#         memo[id(self)] = result
#         for k, v in self.__dict__.items():
#             if k == 'shape_feat':
#                 setattr(result, k, None)  
#             else:
#                 setattr(result, k, copy.deepcopy(v, memo))
#         return result