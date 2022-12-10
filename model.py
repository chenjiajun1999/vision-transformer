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



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=768, num_heads=12, drop=0.):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.attn_drop = DropPath(drop)
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


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MLP(nn.Sequential):
    def __init__(self, embed_size, expansion=4, drop=0.):
        super().__init__(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            DropPath(drop),
            nn.Linear(expansion * embed_size, embed_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 embed_size=768,
                 drop=0.1,
                 expansion=4,
                 **kwargs):
        super().__init__(
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(embed_size),
                MultiHeadAttention(embed_size=embed_size, drop=drop, **kwargs),
                DropPath(drop)
            )),
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(embed_size),
                MLP(
                    embed_size=embed_size, expansion=expansion, drop=drop),
                DropPath(drop)
            )))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassifierHead(nn.Sequential):
    def __init__(self, embed_size=768, num_class=1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_class))


class ViT(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768,
                 img_size=224, depth=12, num_class=1000, **kwargs):
        super().__init__(
            PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_size=embed_size),
            TransformerEncoder(depth=depth, embed_size=embed_size, **kwargs),
            ClassifierHead(embed_size=embed_size, num_class=num_class)
        )


def create_model(num_class=1000):
    return ViT(num_class=num_class)
