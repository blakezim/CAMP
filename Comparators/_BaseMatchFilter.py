import torch.nn as nn


class Filter(nn.Module):
    def __init__(self, source, target, similarity, regularization=None, smoothing=None):
        super(Filter, self).__init__()

        self.source = source
        self.target = target
        self.similarity = similarity
        self.regularization = regularization
        self.smoothing = smoothing

    def forward(self, x):
        return x

    def energy(self):
        raise NotImplementedError

    def update(self, field):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
