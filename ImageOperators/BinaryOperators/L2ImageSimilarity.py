import torch

from ImageOperators.UnaryOperators.GradientFilter import Gradient
from ._BaseFilter import Filter


class L2Similarity(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(L2Similarity, self).__init__()

        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        filt = L2Similarity(dim, device, dtype)
        filt = filt.to(device)  # This will move all tensors in the register_buffer
        filt = filt.type(dtype)

        # Can't add Field and Images to the register buffer, so we need to make sure they are on the right device
        for attr, val in filt.__dict__.items():
            if hasattr(val, 'to'):
                val.to(device)
            if hasattr(val, 'to_'):
                val.to_(device)
            if hasattr(val, 'type'):
                val.type(dtype)
            else:
                pass

        return filt

    def forward(self, target, moving):
        return (target - moving) ** 2

    def c1(self, target, moving):
        grads = self.gradient_operator(moving)
        return ((target - moving) * grads) * 0.5
