import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize

from model import MultiHeadAttention, PatchEmbedding

img = nn.Parameter(torch.randn(3, 1024, 960))
# resize to imagenet size
transform = Compose([Resize((224, 224))])
x = transform(img)
x = x.unsqueeze(0)  # add batch dim
x = PatchEmbedding()(x)
x = MultiHeadAttention()(x)
print(x.shape)
