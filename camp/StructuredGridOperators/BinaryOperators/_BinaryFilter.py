import torch.nn as nn


class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()

    def forward(self, **kwargs):
        raise NotImplementedError

    def c1(self, **kwargs):
        return False
