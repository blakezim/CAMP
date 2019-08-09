import torch

from ImageOperators.UnaryOperators.GradientFilter import Gradient
from ._BinaryFilter import Filter


class L2Similarity(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(L2Similarity, self).__init__()

        self.device = device
        self.dtype = dtype
        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        filt = L2Similarity(dim, device, dtype)
        filt = filt.to(device)
        filt = filt.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in filt.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return filt

    def forward(self, target, moving):
        return 0.5 * ((target - moving) ** 2)

    def c1(self, target, moving, field):
        grads = self.gradient_operator(moving)
        return (target - moving) * grads
