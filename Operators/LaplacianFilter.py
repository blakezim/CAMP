import math
import torch
import torch.nn.functional as F

from ._BaseFilter import Filter


class Laplacian(Filter):
    def __init__(self, size, dim=2, gamma=0.001, alpha=1.0, incompresible=False, device='cpu'):
        super(Laplacian, self).__init__()

        # Create the vectors to mesh grid
        vecs = [torch.arange(0, x).float() for x in size]

        # Create the grids from the vectors
        grids = torch.meshgrid(vecs)

        # Concatenate the grids in the first dimension
        grids = [grid.view(1, *grid.shape) for grid in grids]
        self.grids = torch.cat(grids, 0)

        if dim == 2:
            self.H = -4 + 2 * torch.cos((2 * math.pi * self.grids[0]) / (size[0])) + \
                     2 * torch.cos((2 * math.pi * self.grids[1]) / (size[1]))
            self.H = self.H

        self.H = -(self.H * alpha) + gamma
        self.H = self.H.reciprocal().to(device)

    def forward(self, grads):

        # Take the fourier transform of the data
        fft = torch.rfft(grads.double(), signal_ndim=2, normalized=False, onesided=False)
        for g in range(0, 2):
            for c in range(0, 2):
                fft[g, :, :, c] = fft[g, :, :, c] * self.H.double()
        # fft *= self.H.double()
        grads = torch.irfft(fft, signal_ndim=2, normalized=False, onesided=False)
        return grads.float()
