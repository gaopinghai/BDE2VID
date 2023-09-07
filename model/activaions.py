import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Registry

ACTIVATION = Registry('activation')


@ACTIVATION.register_module()
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

@ACTIVATION.register_module()
class LReLU(nn.Module):
    def __init__(self, negative_slope=1e-2, inplace=True):
        super(LReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope, self.inplace)