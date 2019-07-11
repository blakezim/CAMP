import math
import torch

from Core.ImageClass import Image
from Core.GridClass import Grid
from ._BaseFilter import Filter


class InverseLaplacian(Filter):
    def __init__(self, size, gamma=0.001, alpha=1.0, incompresible=False):
        super(InverseLaplacian, self).__init__()

        self.incompressible = incompresible

        # Create the vectors to mesh grid
        vecs = [torch.arange(0, x).float() for x in size]

        # Create the grids from the vectors
        grids = torch.meshgrid(vecs)

        # Concatenate the grids in the first dimension
        grids = [grid.view(1, *grid.shape) for grid in grids]
        grids = torch.cat(grids, 0)

        # Take the cosine of the grids
        cos_grids = [2 * torch.cos((2 * math.pi * grids[d])/size[d]).view(1, *grids[d].shape) for d in range(len(size))]
        cos_grids = torch.cat(cos_grids, 0)

        # Sum the cosine grids and add the center term
        H = (-2 * len(size)) + cos_grids.sum(0)
        H = -(H * alpha) + gamma
        H = H.reciprocal()
        H = H.view(1, *H.shape, 1)

        # Add the inverse laplacian to the register_buffer
        self.register_buffer('H', H)

    @staticmethod
    def Create(size, gamma=0.001, alpha=1.0, incompresible=False, device='cpu', dtype=torch.float32):
        lap = InverseLaplacian(size, gamma, alpha, incompresible)
        lap = lap.to(device)
        lap = lap.type(dtype)
        return lap

    def forward(self, x):

        out = x.clone()

        if type(x).__name__ == 'Tensor':
            out = Image.FromGrid(
                Grid(x.shape[1:],
                     device=self.field.device,
                     dtype=self.field.dtype,
                     requires_grad=self.field.requires_grad),
                tensor=x.clone(),
                channels=x.shape[0]
            )

        # Take the fourier transform of the data
        fft = torch.rfft(out.data, signal_ndim=2, normalized=False, onesided=False)

        # Apply the filter
        fft *= self.H

        # Inverse fourier transform
        out.data = torch.irfft(fft, signal_ndim=2, normalized=False, onesided=False)

        return out
