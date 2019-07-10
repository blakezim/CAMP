import torch.nn as nn


class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def c1():
        return False
