import torch.nn as nn


class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()

    def forward(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def Create(**kwargs):
        raise NotImplementedError
