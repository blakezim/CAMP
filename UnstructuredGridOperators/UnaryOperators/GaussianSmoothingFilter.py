import math
import torch
import numbers
import itertools

import torch.nn.functional as F

from ._UnaryFilter import Filter


class GaussianSmoothing(Filter):
    def __init__(self, sigma, dim=2, device='cpu', dtype=torch.float32):
        super(GaussianSmoothing, self).__init__()

        self.device = device
        self.dtype = dtype

        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.register_buffer('sigma', sigma)


    @staticmethod
    def Create(sigma, dim=2, device='cpu', dtype=torch.float32):
        gauss = GaussianSmoothing(sigma, dim, device, dtype)
        gauss = gauss.to(device=device, dtype=dtype)

        return gauss

    def forward(self, verts):
        grads = verts.grad

        # Need to calculate the distance between all other points
        d = ((verts.unsqueeze(1) - verts.unsqueeze(0)) ** 2).sum(-1, keepdim=True)

        out = (grads[None, :, :].repeat(len(grads), 1, 1) * torch.exp(-d / (2*self.sigma[None, None, :]))).sum(1)

        return out
