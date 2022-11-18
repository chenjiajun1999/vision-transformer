import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_size),
        )

        # 生成一个维度为 embed_size 的向量当做 cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

        # 位置编码信息，一共有 (img_size // patch_size)**2 + 1(cls token) 个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_size))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.projection(x)
        # 将cls_token 扩展 B 次
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=768, num_heads=8, dropout=0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # [batch_size, num_patches + 1, embed_size]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * embed_size]
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scaling = self.embed_size ** (1 / 2)
        # batch, num_heads, query_len, key_len
        attn = torch.einsum('bhqd, bhkd -> bhqk', q, k) / scaling
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # batch, num_heads, attn_len, value_len
        x = torch.einsum('bhal, bhlv -> bhav ', attn, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.projection(x)
        return x
