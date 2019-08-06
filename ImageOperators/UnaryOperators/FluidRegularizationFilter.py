import torch

from .FluidKernelFilter import FluidKernel
from ._UnaryFilter import Filter


class FluidRegularization(Filter):
    def __init__(self, grid, alpha=1.0, beta=0.0, gamma=0.001, inverse=True,
                 incompresible=False, device='cpu', dtype=torch.float32):
        super(FluidRegularization, self).__init__()

        self.device = device
        self.dtype = dtype

        self.identity = grid.clone()
        self.identity.set_to_identity_lut_()
        self.fluid_operator = FluidKernel(grid, alpha, beta, gamma, inverse, incompresible, device, dtype)

    @staticmethod
    def Create(grid, alpha=1.0, beta=0.0, gamma=0.001, inverse=True,
               incompresible=False, device='cpu', dtype=torch.float32):
        reg = FluidRegularization(grid, alpha, beta, gamma, inverse, incompresible, device, dtype)
        reg = reg.to(device)
        reg = reg.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in reg.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return reg

    def forward(self, x):

        # Apply the forward operator
        self.fluid_operator.inverse = False
        x = self.fluid_operator(x)

        return 0.5 * (x ** 2)

    def c1(self, x):

        # Apply the forward operator
        self.fluid_operator.inverse = False
        x = self.fluid_operator(x)

        # Apply the inverse operator
        self.fluid_operator.inverse = True
        x = self.fluid_operator(x)

        return x
