import torch.nn as nn


class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()

    def forward(self, target, moving):
        return target - moving

    def c1(self, target, moving):
        return False
