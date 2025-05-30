import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
# from pytorch_memlab import profile, profile_every

# from mmcv.runner import load_checkpoint
from mmengine.model import BaseModule

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(BaseModule):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU(), drop=0., init_cfg=None):
        super().__init__(init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size, dilate_win):
    """
    Args:
        x: (D, B, C, H, W)
        window_size (tuple[int]): window size
        dilate_win (bool):

    Returns:
        windows: (D, B*num_windows, C, window_size, window_size)
    """
    D, B, C, H, W = x.shape
    if not dilate_win:
        x = x.view(D, B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(D, -1, C, window_size[0], window_size[1])
    else:
        x = x.view(D*B, C, H, W)
        h, w = window_size[-2:]
        x = F.pad(x, (0, w, 0, h))
        x = F.unfold(x, kernel_size=window_size, dilation=(2, 2), stride=window_size, padding=0)
        x = x.permute(0, 2, 1).contiguous().view(D, -1, C, window_size[0], window_size[1])
    return x  # D, B*num_windows, C, window_size, window_size


def window_reverse(windows, B, H, W, dilate_win):
    """
    Args:
        windows: (B*num_windows, C, window_size, window_size)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B_, C, Hw, Ww = windows.shape
    if not dilate_win:
        x = windows.view(B, H//Hw, W//Ww, C, Hw, Ww).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
    else:
        x = windows.view(B, H//Hw * W//Ww, -1).permute(0, 2, 1).contiguous()
        x = F.fold(x, (H+Hw, W+Ww), kernel_size=(Hw, Ww), stride=(Hw, Ww),
                   padding=0, dilation=(2, 2))
        x = x[:, :, :-Hw, :-Ww]
    return x


def get_window_size(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]

    return tuple(use_window_size)


class WindowAttention3D(BaseModule):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, nwin_size, num_heads, qkv_bias=False, qk_scale=None,
                 q_ind=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm,
                 dilate_win=False, init_cfg=None):
        super().__init__(init_cfg)
        q_ind = window_size[0]//2 if q_ind is None else q_ind
        self.q_ind = q_ind
        assert q_ind >= 0
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.nwin_size = nwin_size
        self.dilate_win = dilate_win
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ind_WinS = q_ind * window_size[1] * window_size[2]
        self.q_ind_WinE = self.q_ind_WinS + window_size[1] * window_size[2]

        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        # feature reduction
        if nwin_size is not None:
            self.reduction_conv = nn.Conv2d(dim, reduce(mul, nwin_size+(dim,)), kernel_size=window_size[-2:], groups=dim)
        else:
            self.reduction_conv = None

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    # @profile_every(1)
    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (D, num_windows*B, C, Wh, Ww)
        """
        D, B_, C, H, W = x.shape

        if self.reduction_conv is not None:
            kv = x.view(-1, C, H, W)
            kv = self.reduction_conv(kv)  # -1, C*X, 1, 1
            kv = kv.view(D, B_, self.nwin_size[0] * self.nwin_size[1], C)
        else:
            kv = x.permute(0, 1, 3, 4, 2).contiguous().view(D, B_, H*W, C)
        D, B_, _, C = kv.shape

        q = x[self.q_ind]  # B_, C, H, W
        q = q.permute(0, 2, 3, 1).contiguous().view(B_, -1, C)
        kv = kv.permute(1, 0, 2, 3).contiguous().view(B_, -1, C)
        q = self.norm_q(q)  # B_, H*W, C
        kv = self.norm_kv(kv)  # B_, D*Hn*Wn, C

        M = q.shape[1]
        B_, N, C = kv.shape
        q = self.q(q).reshape(B_, M, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)  # B_, nH, M, C
        kv = self.kv(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, nH, M, N

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[self.q_ind_WinS:self.q_ind_WinE, :N].reshape(-1)].reshape(
            M, N, -1)  # Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn += relative_position_bias.unsqueeze(0)  # B_, nH, M, N
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # B_, nH, M, N
        x = (attn @ v).transpose(1, 2).reshape(B_, M, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B_, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x  # B_, C, H, W


class SwinTransformerBlock3D(BaseModule):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(3, 8, 8), nwindow_size=(3, 3), dilate_win=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU(), norm_layer=nn.LayerNorm, use_checkpoint=False,
                 q_ind=None, init_cfg=None):
        super().__init__(init_cfg)
        q_ind = window_size[0] // 2 if q_ind is None else q_ind
        self.q_ind = q_ind
        assert self.q_ind >= 0
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilate_win = dilate_win
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, nwin_size=nwindow_size, num_heads=num_heads, q_ind=q_ind,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, dilate_win=dilate_win)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # @profile_every(1)
    def forward_part1(self, x):
        D, B, C, H, W = x.shape
        window_size = get_window_size((H, W), self.window_size[-2:])

        # x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_t, pad_b = pad_h//2, pad_h - pad_h//2
        pad_l, pad_r = pad_w//2, pad_w - pad_w//2
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, _, Hp, Wp = x.shape

        # partition windows
        x_windows = window_partition(x, window_size, self.dilate_win)  # D, B*nW, C, Wh, Ww
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # B*nW, C, Wh, Ww
        # merge windows
        x = window_reverse(attn_windows, B, Hp, Wp, self.dilate_win)  # B, C, H, W

        if pad_h > 0 or pad_w > 0:
            _, _, H, W = x.shape
            x = x[:, :, pad_t:H-pad_b, pad_l:W-pad_r].contiguous()
        return x

    def forward_part2(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x

    # @profile_every(1)
    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (D, B, C, H, W).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x[self.q_ind]
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class DFrameAttention(BaseModule):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 nwindow_size=(3, 3),
                 q_ind=None,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 activation="default",
                 init_cfg=None):
        super().__init__(init_cfg)
        activation = "GELU" if activation == "default" else activation
        act_layer = getattr(torch.nn, activation)()
        q_ind = window_size[0] // 2 if q_ind is None else q_ind
        self.q_ind = q_ind
        assert self.q_ind >= 0
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                nwindow_size=nwindow_size,
                q_ind=q_ind,
                dilate_win=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                act_layer=act_layer
            )
            for i in range(depth)])

    # @profile_every(1)
    def forward(self, x):
        """ Forward function.

        Args:
            x (list): Input feature, [feat]*D, feat: B, C, H, W.
        """
        # calculate attention mask for SW-MSA
        keys = x  # D = len(keys)
        x = keys[self.q_ind]  # B, C, H, W
        for i, blk in enumerate(self.blocks):
            keys[self.q_ind] = x
            x = torch.stack(keys, dim=0)  # D, B, C, H, W
            x = blk(x)
        return x  # B, C, H, W

