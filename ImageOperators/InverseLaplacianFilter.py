from math import pi

import torch

from Core.ImageClass import Image
from Core.GridClass import Grid
from ._BaseFilter import Filter

#
# def roll_n(X, axis, n):
#     f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
#     b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
#     front = X[f_idx]
#     back = X[b_idx]
#     return torch.cat([back, front], axis)
#
#
# def batch_fftshift2d(x):
#     # real, imag = torch.unbind(x, -1)
#     for dim in range(1, len(x.size())):
#         n_shift = real.size(dim)//2
#         if real.size(dim) % 2 != 0:
#             n_shift += 1  # for odd-sized images
#         real = roll_n(real, axis=dim, n=n_shift)
#         imag = roll_n(imag, axis=dim, n=n_shift)
#     return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
#
#
# def fftshift2d(x):
#     for dim in range(0, len(x.size())):
#         n_shift = x.size(dim) // 2
#         if x.size(dim) % 2 != 0:
#             n_shift += 1  # for odd-sized images
#         x = roll_n(x, axis=dim, n=n_shift)
#     return x


class InverseLaplacian(Filter):
    def __init__(self, grid, alpha=1.0, beta=0.0, gamma=0.001, incompresible=False, device='cpu'):
        super(InverseLaplacian, self).__init__()

        self.incompresible = incompresible
        self.device = device
        # Expand the size and spacing variable so they are easily usable
        size = grid.size.view(*grid.size.shape, *([1] * len(grid.size)))
        spacing = grid.spacing.view(*grid.spacing.shape, *([1] * len(grid.spacing)))

        # Create the vectors to mesh grid
        vecs = [torch.arange(0, x, device=device).float() for x in grid.size]

        # Create the grids from the vectors and stack them in the first dimension
        # vecs[0], vecs[1] = vecs[1], vecs[0]
        grids = list(torch.meshgrid(vecs))
        grids[-2], grids[-1] = grids[-1], grids[-2]
        grids = torch.stack(grids, 0)

        # Compute the cosine grids
        cos_grids = (2 * torch.cos((2.0 * pi * grids) / size) - 2) / (spacing * spacing)
        sin_grids = torch.sin((2 * pi * grids) / size) / spacing

        # Make the spatial dimensions linear
        cos_grids = cos_grids.view(len(grid.size), -1)
        sin_grids = sin_grids.view(len(grid.size), -1)

        # Now that we have the cosine and sine grids, we can compute L
        lmbda = (-alpha * cos_grids.sum(0) + gamma)
        L = torch.eye(len(grid.size), device=device).unsqueeze(-1).repeat(1, 1, len(lmbda))

        # Fill in the diagonal terms
        for x in range(0, len(grid.size)):
            for y in range(0, len(grid.size)):
                if x == y:
                    L[x, y, :] = lmbda - (beta * cos_grids[x])
                else:
                    L[x, y, :] = (beta * sin_grids[x] * sin_grids[y])

        # Do the cholskey decomposition on the CPU - GPU is not working
        L = L.permute(2, 0, 1).cpu()
        G = torch.cholesky(L)
        G = G.to(self.device)

        # Go ahead and compute the norm squared of the sin_grids
        nsq = (sin_grids ** 2).sum(0)

        # Add variable to the buffer so we ensure that they are on the right device
        self.register_buffer('nsq', nsq)
        self.register_buffer('sin_grids', sin_grids.permute(1, 0))
        self.register_buffer('cos_grids', cos_grids.permute(1, 0))
        self.register_buffer('G', G)
        # self.register_buffer('test', test)

    def solve_cholskey(self, x):
        # back-solve Gt = x to get a temporary vector t
        # back-solve G'y = t to get answer in y
        # [G(0, 0)     0       0  ]     [y(1)] = b(1)
        # [G(1, 0) G(1, 1)     0  ]  *  [y(2)] = b(2)
        # [G(2, 0) G(2, 1) G(2, 2)]     [y(3)] = b(3)
        #
        # [G(0, 0) G(1, 0) G(2, 0)]     [x(1)] = y(1)
        # [   0    G(1, 1) G(2, 1)]  *  [x(2)] = y(2)
        # [   0       0    G(2, 2)]     [x(3)] = y(3)

        t0 = x[:, 0] / self.G[:, 0, 0]
        t1 = (x[:, 1] - (self.G[:, 1, 0] * t0)) / self.G[:, 1, 1]

        if x.shape[-1] == 3:
            t2 = (x[:, 1] - (self.G[:, 2, 0] * t0) - (self.G[:, 2, 1] * t1)) / self.G[:, 2, 2]
            y2 = t2 / self.G[:, 2, 2]
            y1 = (t1 - (self.G[:, 2, 1] * y2)) / self.G[:, 1, 1]
            y0 = (t0 - (self.G[:, 1, 0] * y1) - (self.G[:, 2, 0] * y2)) / self.G[:, 0, 0]

            y = torch.stack((y0, y1, y2), -1)

        else:
            y1 = t1 / self.G[:, 1, 1]
            y0 = (t0 - (self.G[:, 1, 0] * y1)) / self.G[:, 0, 0]
            y = torch.stack((y0, y1), -1)

        return y

    @staticmethod
    def Create(grid, alpha=1.0, beta=0.0, gamma=0.001, incompresible=False, device='cpu', dtype=torch.float32):
        lap = InverseLaplacian(grid, alpha, beta, gamma, incompresible, device)
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
        real, imag = fft.split(1, -1)

        # We need to linearize the matrices so we can easily do the multiplication
        real = real.squeeze().view(len(x.size), -1).permute(1, 0)
        imag = imag.squeeze().view(len(x.size), -1).permute(1, 0)

        real_smooth = self.solve_cholskey(real)
        imag_smooth = self.solve_cholskey(imag)

        if self.incompresible:

            real_dot = (real_smooth * self.sin_grids).sum(-1, keepdim=True)
            imag_dot = (imag_smooth * self.sin_grids).sum(-1, keepdim=True)

            real_smooth = real_smooth - (real_dot * self.sin_grids / self.nsq.unsqueeze(-1))
            imag_smooth = imag_smooth - (imag_dot * self.sin_grids / self.nsq.unsqueeze(-1))

            # Deal with the nan
            real_smooth[real_smooth != real_smooth] = real.squeeze()[real_smooth != real_smooth]
            imag_smooth[imag_smooth != imag_smooth] = imag.squeeze()[imag_smooth != imag_smooth]

        # # now return them to their actual shapes
        real_smooth = real_smooth.permute(1, 0).view(len(x.size), *x.size.long())
        imag_smooth = imag_smooth.permute(1, 0).view(len(x.size), *x.size.long())
        smooth_fft = torch.stack((real_smooth, imag_smooth), -1)

        # Inverse fourier transform
        out.data = torch.irfft(smooth_fft, signal_ndim=2, normalized=False, onesided=False)

        return out
