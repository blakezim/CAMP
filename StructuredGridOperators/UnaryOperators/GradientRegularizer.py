import torch

from StructuredGridOperators.UnaryOperators.GradientFilter import Gradient
from ._UnaryFilter import Filter


class NormGradient(Filter):
    def __init__(self, weight, dim=2, device='cpu', dtype=torch.float32):
        super(NormGradient, self).__init__()

        self.weight = weight
        self.device = device
        self.dtype = dtype
        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    @staticmethod
    def Create(weight, dim=2, device='cpu', dtype=torch.float32):
        filt = NormGradient(weight, dim, device, dtype)
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

    def forward(self, vector_field):
        return self.weight * (self.gradient_operator(vector_field) ** 2)
