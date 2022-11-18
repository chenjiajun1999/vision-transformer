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
        self.num_patch = self.grid_size[0] * self.grid_size[1]

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

