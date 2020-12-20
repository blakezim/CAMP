import torch
import itertools
import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from .GradientFilter import Gradient
from ._UnaryFilter import Filter


class Divergence(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(Divergence, self).__init__()

        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.gradient = Gradient(dim=dim)
        self.pad = F.pad
        self.pad_vec = []
        self.pad_vec += [[1, 1] for _ in range(0, dim)]
        self.pad_vec = tuple(itertools.chain.from_iterable(self.pad_vec))

        self.crop_vec = []
        self.crop_vec += [[-1, -1] for _ in range(0, dim)]
        self.crop_vec = tuple(itertools.chain.from_iterable(self.crop_vec))

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        div = Divergence(dim, device, dtype)
        div = div.to(device)
        div = div.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in div.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return div

    def forward(self, x):

        grads = self.gradient(x)
        div = grads[torch.eye(self.dim, dtype=torch.bool).flatten(), :, :].sum(dim=0, keepdim=True)

        out = StructuredGrid.FromGrid(
            x,
            tensor=div,
            channels=1
        )

        return out
