import torch
import math
import torch.nn as nn


class MaxPooling(nn.Module):
    def __init__(self, dim):
        super(MaxPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, self.dim)[0]


class SumPooling(nn.Module):
    def __init__(self, dim):
        super(SumPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, self.dim)
