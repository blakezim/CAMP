import torch

from ._UnaryFilter import Filter


class FluidRegularization(Filter):
    def __init__(self, device='cpu', dtype=torch.float32):
        super(FluidRegularization, self).__init__()

        self.device = device
        self.dtype = dtype

    @staticmethod
    def Create(device='cpu', dtype=torch.float32):
        reg = FluidRegularization(device, dtype)
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

    @staticmethod
    def forward(x, operator):

        # Apply the forward operator
        x = operator.apply_forward(x)
        return 0.5 * (x ** 2)

    @staticmethod
    def c1(x, operator):

        # Apply the forward operator
        x = operator.apply_forward(x)

        # Apply the inverse operator
        x = operator.apply_inverse(x)

        return x
