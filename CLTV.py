# -*- coding: utf-8 -*-


class ImageLinearAttention(nn.Module):
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2d(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
            k = torch.cat((k, ck), dim=3)
            v = torch.cat((v, cv), dim=3)

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
    
class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b



class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head = 32, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                ImageLinearAttention(chan = dim, chan_out = None, kernel_size = 3, padding = 1, stride = 1, key_dim = 32, value_dim = 32, heads = heads, norm_queries = True),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CLTV(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        emb_dim = 64,
        emb_kernel = 7,
        emb_stride = 4,
        heads = 1,
        depth = 1,
        mlp_mult = 4,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = int(emb_dim/2)
        self.conv = nn.Sequential(
            nn.Conv2d(3, int(dim/2), 3, 1, 1),
            nn.Conv2d(int(dim/2), dim, 3, 1, 1)
        )
                
        
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim, emb_dim, kernel_size = emb_kernel, padding = emb_kernel// 2, stride = emb_stride),
                LayerNorm(emb_dim),
                Transformer(dim = emb_dim, heads = heads, mlp_mult = mlp_mult, dropout = dropout)
            ))


            dim = emb_dim

        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)

        for cnn, norm, transformer in self.layers:
            x = cnn(x)
            x = norm(x)
            x = transformer(x)


        
        return self.head(x)

model = CLTV(
    num_classes = 10,
    emb_dim = 128,        # stage 1 - dimension
    emb_kernel = 3,      # stage 1 - conv kernel
    emb_stride = 1,      # stage 1 - conv stride
    heads = 4,           # stage 1 - heads
    depth = 5,           # stage 1 - depth
    mlp_mult = 2,        # stage 1 - feedforward expansion factor
    dropout = 0.5
)
