import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize

from model import ViT

img = nn.Parameter(torch.randn(3, 1024, 960))
# resize to imagenet size
transform = Compose([Resize((224, 224))])
x = transform(img)
x = x.unsqueeze(0)  # add batch dim
x = ViT()(x)

print(x.shape)
