"""
Core of BiFormer, Bi-Level Routing Attention.

To be refactored.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, LongTensor


"""
Deformable Region-Level BRA: 在区域级路由阶段引入可变形区域（单组变形假设版）
取消多组变形假设，使用单一变形假设
修复了所有已知的内存连续性和维度错误
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, LongTensor
from typing import Tuple, Optional


class DeformableRegionBRA(nn.Module):
    """
    在BRA的区域级路由阶段引入可变形区域
    取消多组变形假设，使用单一变形假设
    """
    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, 
                 kv_downsample_mode='identity', topk=4, param_attention="qkvo", 
                 param_routing=False, diff_routing=True, soft_routing=True,
                 side_dwconv=3, num_deform_points=12, auto_pad=True):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.topk = topk
        self.num_deform_points = num_deform_points
        
        # 可变形区域参数 - 移除了deform_groups参数，固定为1组
        self.region_offset_conv = nn.Conv2d(
            self.qk_dim * 2,  # 输入通道：查询+键特征拼接
            2 * num_deform_points,  # 输出通道：每个点的(x,y)偏移，只有1组
            kernel_size=1,
            stride=1,
            padding=0
        )
        # nn.init.constant_(self.region_offset_conv.weight, 0)
        # nn.init.constant_(self.region_offset_conv.bias, 0)
        
        # 区域重要性权重预测 - 移除了deform_groups参数
        self.region_weight_conv = nn.Conv2d(
            self.qk_dim * 2,
            num_deform_points,  # 输出通道：每个点的权重，只有1组
            kernel_size=1,
            stride=1,
            padding=0
        )
        # nn.init.constant_(self.region_weight_conv.weight, 0)
        # nn.init.constant_(self.region_weight_conv.bias, 0)
        
        # 标准BRA组件
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, 
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        
        # 修改路由器以支持可变形区域（使用单组版本）
        self.router = DeformableTopkRouting_SingleGroup(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=diff_routing,
            param_routing=param_routing,
            num_deform_points=num_deform_points
        )
        
        self.kv_gather = KVGather(mul_weight='soft' if soft_routing else 'none')
        
        # QKV映射
        if param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {param_attention} is not supported!')
        
        # KV下采样
        self.kv_downsample_mode = kv_downsample_mode
        if kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(kv_per_win)
        elif kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(kv_per_win)
        elif kv_downsample_mode == 'maxpool':
            self.kv_down = nn.MaxPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        else:
            raise ValueError(f'kv_down_sample_mode {kv_downsample_mode} is not supported!')
        
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad
    
    def deformable_region_sampling(self, feature_map: Tensor, offsets: Tensor, 
                                  weights: Tensor, region_size: tuple):
        """
        可变形区域采样（单组版本）
        Args:
            feature_map: (B, C, H, W) 特征图
            offsets: (B, num_deform_points, 2, H, W) 偏移量
            weights: (B, num_deform_points, H, W) 采样点权重
            region_size: (h, w) 基础区域大小
        Returns:
            region_features: (B, C, H, W) 采样得到的区域特征
        """
        B, C, H, W = feature_map.shape
        _, num_points, _, H_o, W_o = offsets.shape
        
        # 生成基础采样网格
        y, x = torch.meshgrid(torch.arange(H_o), torch.arange(W_o), indexing='ij')
        base_grid = torch.stack([x, y], dim=-1).float().to(feature_map.device)  # (H_o, W_o, 2)
        base_grid = base_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H_o, W_o, 2)
        base_grid = base_grid.repeat(B, num_points, 1, 1, 1)
        
        # 添加偏移
        offsets = offsets.permute(0, 1, 3, 4, 2)  # (B, P, H, W, 2)
        sample_grid = base_grid + offsets
        
        # 归一化到[-1, 1]
        sample_grid[..., 0] = 2.0 * sample_grid[..., 0] / (W - 1) - 1.0
        sample_grid[..., 1] = 2.0 * sample_grid[..., 1] / (H - 1) - 1.0
        
        # 采样特征
        sampled_features = []
        for p in range(num_points):
            # 当前采样点的网格
            grid = sample_grid[:, p]  # (B, H_o, W_o, 2)
            
            # 双线性插值采样
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # (B, C, H_o, W_o)
            sampled_features.append(sampled)
        
        # 加权融合多个采样点
        sampled_features = torch.stack(sampled_features, dim=1)  # (B, P, C, H_o, W_o)
        weights_expanded = weights.unsqueeze(2)  # (B, P, 1, H_o, W_o)
        weighted_features = (sampled_features * weights_expanded).sum(dim=1)  # (B, C, H_o, W_o)
        
        return weighted_features
    
    def forward(self, x, ret_attn_mask=False):
        # 确保输入是NCHW格式，并重新排列为NHWC格式
        if x.dim() == 4 and x.size(1) == self.dim:
            x = rearrange(x, "n c h w -> n h w c")
        
        N, H_in, W_in, C = x.size()
        
        if self.auto_pad:
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        N, H, W, C = x.size()
        assert H % self.n_win == 0 and W % self.n_win == 0
        
        # 分窗口
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", 
                      j=self.n_win, i=self.n_win, h=H//self.n_win, w=W//self.n_win)
        
        # QKV投影
        q, kv = self.qkv(x)
        
        # 像素级QKV
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        
        # KV下采样
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', 
                          j=self.n_win, i=self.n_win)
        
        # === 可变形区域特征提取 ===
        # 准备查询和键特征用于预测区域参数
        q_feat = rearrange(q, 'n p2 h w c -> (n p2) c h w')
        k_feat = rearrange(kv[..., :self.qk_dim], 'n p2 h w c -> (n p2) c h w')
        
        # 拼接查询和键特征
        qk_feat = torch.cat([q_feat, k_feat], dim=1)  # (n*p2, 2*C, h, w)
        
        # 预测偏移量和权重（单组版本）
        offsets = self.region_offset_conv(qk_feat)
        
        # 使用 reshape 而不是 view 避免内存连续性问题
        offsets = offsets.reshape(N * (self.n_win ** 2), self.num_deform_points, 2, 
                                 H // self.n_win, W // self.n_win)
        
        weights = self.region_weight_conv(qk_feat)
        weights = weights.reshape(N * (self.n_win ** 2), self.num_deform_points, 
                                 H // self.n_win, W // self.n_win)
        weights = F.softmax(weights, dim=1)  # 归一化采样点权重
        
        # === 修复的关键部分：使用 reshape 而不是 view ===
        # 获取 kv 的形状信息
        n, p2, h, w, c_total = kv.shape
        
        # 验证维度
        assert c_total == self.qk_dim + self.dim, \
            f"Expected total channels {self.qk_dim + self.dim}, got {c_total}"
        
        # 修复：使用 reshape 处理可能不连续的内存
        kv_feature_map = kv.reshape(
            N * (self.n_win ** 2),        # (n p2)
            self.qk_dim + self.dim,       # 总通道数
            H // self.n_win,              # h
            W // self.n_win               # w
        )
        
        # 可变形区域采样（单组版本）
        deformable_kv = self.deformable_region_sampling(
            kv_feature_map,
            offsets,
            weights,
            region_size=(H // self.n_win, W // self.n_win)
        )  # (n*p2, C_kv, h, w)
        
        # 重塑为窗口格式 - 再次使用 reshape
        deformable_kv = deformable_kv.reshape(N, self.n_win ** 2, 
                                             H // self.n_win, W // self.n_win, 
                                             self.qk_dim + self.dim)
        
        # 确保正确的维度顺序
        deformable_kv = rearrange(deformable_kv, 'n (j i) h w c -> n (j i) h w c', 
                                 j=self.n_win, i=self.n_win)
        
        # 从可变形区域提取窗口级特征
        q_win = q.mean([2, 3])
        k_win_deform = deformable_kv[..., :self.qk_dim].mean([2, 3])
        # k_win_original = kv[..., :self.qk_dim].mean([2, 3])
        
        # 融合原始和可变形特征
        # k_win = 0.5 * k_win_original + 0.5 * k_win_deform
        k_win = k_win_deform
        
        # LCE（局部上下文增强）
        lepe_input = rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', 
                               j=self.n_win, i=self.n_win)
        if not lepe_input.is_contiguous():
            lepe_input = lepe_input.contiguous()
        lepe = self.lepe(lepe_input)
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', 
                        j=self.n_win, i=self.n_win)
        
        # 使用可变形特征进行路由
        r_weight, r_idx = self.router(q_win, k_win)
        
        # 收集KV（使用原始KV，但路由基于可变形特征）
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        
        # 注意力计算
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)
        
        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', 
                        j=self.n_win, i=self.n_win, h=H//self.n_win, w=W//self.n_win)
        
        out = out + lepe
        out = self.wo(out)
        
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :]
            if not out.is_contiguous():
                out = out.contiguous()
        
        if ret_attn_mask:
            return rearrange(out, "n h w c -> n c h w"), r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


class DeformableTopkRouting_SingleGroup(nn.Module):
    """
    支持可变形区域的路由器（单组版本）
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, 
                 diff_routing=True, num_deform_points=9):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.num_deform_points = num_deform_points
        
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)
        
        # 可变形路由的参数（单组版本）
        # if param_routing:
        #     self.deform_mlp = nn.Sequential(
        #         nn.Linear(qk_dim * 2, qk_dim),
        #         nn.ReLU(),
        #         nn.Linear(qk_dim, num_deform_points * 3),  # 偏移x,y + 权重，只有1组
        #     )
        # else:
        #     self.deform_mlp = None
    
    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        
        query_hat, key_hat = self.emb(query), self.emb(key)
        
        # 计算基础亲和度
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        
        # 如果使用可变形路由，调整亲和度
        # if self.deform_mlp is not None:
        #     # 为每个查询-键对预测可变形参数
        #     B, N, C = query_hat.shape
        #     M = key_hat.shape[1]
            
        #     # 扩展维度以获取所有查询-键对
        #     q_expanded = query_hat.unsqueeze(2).expand(-1, -1, M, -1)  # (B, N, M, C)
        #     k_expanded = key_hat.unsqueeze(1).expand(-1, N, -1, -1)    # (B, N, M, C)
            
        #     qk_feat = torch.cat([q_expanded, k_expanded], dim=-1)
        #     deform_params = self.deform_mlp(qk_feat)
        #     deform_params = deform_params.view(B, N, M, self.num_deform_points, 3)
            
        #     # 使用可变形参数调整亲和度
        #     deform_weights = deform_params[..., 2].mean(dim=-1)  # (B, N, M)
        #     attn_logit = attn_logit + deform_weights * 0.1  # 调整亲和度
        
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        
        return r_weight, topk_index


# =========== 依赖的基础类（从原BRA代码复制） ===========

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        
        # 使用 gather 收集键值对
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv))
        
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        
        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv

