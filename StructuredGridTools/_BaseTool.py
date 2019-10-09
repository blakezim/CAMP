import torch.nn as nn


class Filter(nn.Module):
    def __init__(self, source, target):
        super(Filter, self).__init__()

        self.source = source
        self.target = target

    def forward(self, **kwargs):
        raise NotImplementedError
