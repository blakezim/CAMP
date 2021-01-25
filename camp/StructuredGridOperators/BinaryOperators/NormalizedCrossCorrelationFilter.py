import torch
import numbers
# import functools
import itertools
import torch.nn.functional as F

from ..UnaryOperators.GradientFilter import Gradient
from ..UnaryOperators.JacobianDeterminantFilter import JacobianDeterminant
# from Core.StructuredGridClass import StructuredGrid
from ._BinaryFilter import Filter


class NormalizedCrossCorrelation(Filter):
    def __init__(self, grid, window=5, device='cpu', dtype=torch.float32):
        super(NormalizedCrossCorrelation, self).__init__()

        self.device = device
        self.dtype = dtype
        dim = len(grid.size)

        if isinstance(window, numbers.Number):
            window = tuple([window] * dim)
        else:
            window = tuple(window)

        padding = []
        padding += [[(x - 1) // 2, (x - 1) // 2 + (x - 1) % 2] for x in window]
        padding = tuple(itertools.chain.from_iterable(padding))

        kernel = torch.ones(window)
        kernel = (kernel / kernel.sum())
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(grid.shape()[0], *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.jacobian_operator = JacobianDeterminant(dim=dim, device=device, dtype=dtype)
        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

        if dim == 1:
            self.conv = F.conv1d
            self.padding = torch.nn.ReplicationPad1d(padding)
        elif dim == 2:
            self.conv = F.conv2d
            self.padding = torch.nn.ReplicationPad2d(padding)
        elif dim == 3:
            self.conv = F.conv3d
            self.padding = torch.nn.ReplicationPad3d(padding)
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    @staticmethod
    def Create(grid, window=5, device='cpu', dtype=torch.float32):
        filt = NormalizedCrossCorrelation(grid, window, device, dtype)
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

        I_bar = target  - self.conv(input=self.padding(target.data.unsqueeze(0)), weight=self.weight).squeeze(0)
        J_bar = moving  - self.conv(input=self.padding(moving.data.unsqueeze(0)), weight=self.weight).squeeze(0)

        # cc = ((I_bar * J_bar) ** 2).sum() / ((I_bar ** 2).sum() * (J_bar ** 2).sum()) * (moving.data.numel())
        cc = ((I_bar * J_bar).sum()) ** 2 / ((I_bar ** 2).sum() * (J_bar ** 2).sum())

        return 1.0 / cc

    def c1(self, target, moving, grads):

        I_bar = target  - self.conv(input=self.padding(target.data.unsqueeze(0)), weight=self.weight).squeeze(0)
        J_bar = moving  - self.conv(input=self.padding(moving.data.unsqueeze(0)), weight=self.weight).squeeze(0)

        A = (I_bar * J_bar)
        B = (I_bar ** 2)
        C = (J_bar ** 2)

        scale = ((2*A.sum()) / (B.sum()*C.sum()))

        # grads = self.gradient_operator(J_bar)
        # jacdet = self.jacobian_operator(field)

        out = ((I_bar * grads) - ((J_bar * grads) * (A.sum() / C.sum()))) * scale #* jacdet

        return -1 * out
