import torch

from ._UnaryFilter import Filter


class VectorNorm(Filter):
    def __init__(self, p=2, dim=0, keepdim=True, dtype=torch.float32):
        super(VectorNorm, self).__init__()

        self.power = p
        self.dim = dim
        self.keepdim = keepdim
        self.type = dtype

    def forward(self, x):

        out = x.clone()
        out.data = torch.norm(
            out.data,
            p=self.power,
            dim=self.dim,
            keepdim=self.keepdim,
            dtype=self.type
        )

        return out
