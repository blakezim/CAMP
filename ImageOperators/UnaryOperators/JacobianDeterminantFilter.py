import torch

from Core.StructuredGridClass import StructuredGrid
from .GradientFilter import Gradient
from ._UnaryFilter import Filter


class JacobianDeterminant(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(JacobianDeterminant, self).__init__()

        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.gradient = Gradient(dim=dim)

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        jacb = JacobianDeterminant(dim, device, dtype)
        jacb = jacb.to(device)
        jacb = jacb.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in jacb.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return jacb

    def forward(self, x):

        grads = self.gradient(x)
        # Central difference scale the gradients by the spacing
        grads = grads / (2 * x.spacing.repeat(self.dim).view(self.dim ** 2, *([1] * len(x.size))))

        if self.dim == 2:
            det = grads[0] * grads[3] - grads[1] * grads[2]
        else:
            det = grads[0] * (grads[4] * grads[8] - grads[5] * grads[7]) - \
                  grads[1] * (grads[3] * grads[8] - grads[5] * grads[6]) + \
                  grads[2] * (grads[3] * grads[7] - grads[4] * grads[6])

        out = StructuredGrid.FromGrid(
            x,
            tensor=det.view(1, *det.shape),
            channels=1
        )

        return out
