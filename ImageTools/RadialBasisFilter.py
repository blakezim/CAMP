import torch

from ._BaseFilter import Filter


class RadialBasis(Filter):
    def __init__(self, landmarks, device='cpu', dtype=torch.float32):
        super(RadialBasis, self).__init__()

        # self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    @staticmethod
    def Create(landmarks, device='cpu', dtype=torch.float32):
        filt = RadialBasis(landmarks, device, dtype)
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

    def solve_spline(self):
        raise NotImplementedError

    def forward(self, target, moving):
        return (target - moving) ** 2

    def c1(self, target, moving):
        grads = self.gradient_operator(moving)
        return ((target - moving) * grads) * 0.5
