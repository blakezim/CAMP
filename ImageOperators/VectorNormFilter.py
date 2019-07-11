import torch

from ._BaseFilter import Filter


class VectorNorm(Filter):
    def __init__(self, p=2, dim=0, keepdim=True, dtype=torch.float32):
        super(VectorNorm, self).__init__()

        self.power = p
        self.dim = dim
        self.keepdim = keepdim
        self.type = dtype

    def forward(self, x):

        if type(x).__name__ in ['Image', 'Field']:
            out = x.clone()
            out.data = torch.norm(
                out.data,
                p=self.power,
                dim=self.dim,
                keepdim=self.keepdim,
                dtype=self.type
            )
            return out

        elif type(x).__name__ == 'Tensor':
            out = x.clone()
            out = torch.norm(
                out.data,
                p=self.power,
                dim=self.dim,
                keepdim=self.keepdim,
                dtype=self.type
            )
            return out

        else:
            raise RuntimeError(
                'Data type not understood for Gaussian Filter:'
                f' Received type: {type(x).__name__}.  Must be type: [Image, Field, Tensor]'
            )
