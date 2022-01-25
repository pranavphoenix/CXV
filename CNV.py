# -*- coding: utf-8 -*-

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)
            
        self.convert = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',h = 32, w = 32, p1 = 1, p2 = 1)
 
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
 
        )
#         self.pos_embedding = nn.Parameter(torch.randn(1, 1024, dim))
            
    def forward(self, x, pos_emb = None, mask = None, return_attn = False):
        x = self.to_patch_embedding(x)
        
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
#         x += self.pos_embedding[:, :n]
        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

        

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # print(q.shape)
        # set masked positions to 0 in queries, keys, values
        if exists(pos_emb):
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l


        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking



        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        
        return self.convert(out)

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val


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




class Transformer(nn.Module):
    def __init__(self, dim = 128, heads = 8, dim_head = 32, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        num_landmarks = 64
        pinv_iterations = 6
        residual = True
        residual_conv_kernel = 33
        eps = 1e-8
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                NystromAttention(dim, heads, dim_head = dim_head, num_landmarks =num_landmarks, pinv_iterations =pinv_iterations, residual = residual, residual_conv_kernel = residual_conv_kernel, eps = eps, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CNV(nn.Module):
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

model = CNV(
    num_classes = 10,
    emb_dim = 128,        # stage 1 - dimension
    emb_kernel = 3,      # stage 1 - conv kernel
    emb_stride = 1,      # stage 1 - conv stride
    heads = 4,           # stage 1 - heads
    depth = 5,           # stage 1 - depth
    mlp_mult = 2,        # stage 1 - feedforward expansion factor
    dropout = 0.5
)
