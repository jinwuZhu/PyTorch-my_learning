import torch
from torch import nn
img=torch.arange(start=4*4,end = 4*4 + 4*4).reshape(1,1,4,4)
# 池化核和池化步长均为2
pool=nn.AvgPool2d(kernel_size=2,stride=1)
img_2=pool(img)
print(img)
print(img_2)
