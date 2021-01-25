import torch
from math import pi

from ._UnaryFilter import Filter
from ...Core.StructuredGridClass import StructuredGrid
# TODO make the project incompressible and forward use the same fft data


class FluidKernel(Filter):
    def __init__(self, grid, alpha=1.0, beta=0.0, gamma=0.001, device='cpu', dtype=torch.float32):
        super(FluidKernel, self).__init__()

        self.device = device
        self.dtype = dtype
        self.device = device
        self.dim = len(grid.size)  # Need to know the signal dimensions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Expand the size and spacing variable so they are easily usable
        size = grid.size.view(*grid.size.shape, *([1] * len(grid.size)))
        spacing = grid.spacing.view(*grid.spacing.shape, *([1] * len(grid.spacing)))

        # Create the vectors to mesh grid
        vecs = [torch.arange(0, x, device=device).float() for x in grid.size]

        # Create the grids from the vectors and stack them in the first dimension
        grids = torch.meshgrid(vecs)
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
        for i in range(0, len(grid.size)):
            for j in range(0, len(grid.size)):
                if i == j:
                    L[i, j, :] = lmbda - (beta * cos_grids[i])
                else:
                    L[i, j, :] = (beta * sin_grids[i] * sin_grids[j])

        # Do the cholskey decomposition on the CPU - GPU is not working
        L = L.permute(2, 0, 1).cpu()
        G = torch.cholesky(L)
        G = G.to(self.device)

        # Go ahead and compute the norm squared of the sin_grids
        nsq = (sin_grids ** 2).sum(0)

        # Add variable to the buffer so we ensure that they are on the right device
        self.register_buffer('L', L)
        self.register_buffer('nsq', nsq)
        self.register_buffer('sin_grids', sin_grids.permute(1, 0))
        self.register_buffer('cos_grids', cos_grids.permute(1, 0))
        self.register_buffer('G', G)

    @staticmethod
    def Create(grid, alpha=1.0, beta=0.0, gamma=0.001, device='cpu', dtype=torch.float32):

        lap = FluidKernel(grid, alpha, beta, gamma, device, dtype)
        lap = lap.to(device)
        lap = lap.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in lap.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return lap

    def set_size(self, grid):
        return self.Create(
            grid=grid,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            device=self.device,
            dtype=self.dtype
        )

    def _solve_cholskey(self, x):
        # back-solve Gt = x to get a temporary vector t
        # back-solve G'y = t to get answer in y
        # [G(0, 0)     0       0  ]     [t(0)] = x(0)
        # [G(1, 0) G(1, 1)     0  ]  *  [t(1)] = x(1)
        # [G(2, 0) G(2, 1) G(2, 2)]     [t(2)] = x(2)
        #
        # [G(0, 0) G(1, 0) G(2, 0)]     [y(0)] = t(0)
        # [   0    G(1, 1) G(2, 1)]  *  [y(1)] = t(1)
        # [   0       0    G(2, 2)]     [y(2)] = t(2)

        t0 = x[:, 0] / self.G[:, 0, 0]
        t1 = (x[:, 1] - (self.G[:, 1, 0] * t0)) / self.G[:, 1, 1]

        if x.shape[-1] == 3:
            t2 = (x[:, 2] - (self.G[:, 2, 0] * t0) - (self.G[:, 2, 1] * t1)) / self.G[:, 2, 2]
            y2 = t2 / self.G[:, 2, 2]
            y1 = (t1 - (self.G[:, 2, 1] * y2)) / self.G[:, 1, 1]
            y0 = (t0 - (self.G[:, 1, 0] * y1) - (self.G[:, 2, 0] * y2)) / self.G[:, 0, 0]

            y = torch.stack((y0, y1, y2), -1)

        else:
            y1 = t1 / self.G[:, 1, 1]
            y0 = (t0 - (self.G[:, 1, 0] * y1)) / self.G[:, 0, 0]
            y = torch.stack((y0, y1), -1)

        return y

    def forward(self, x, inverse):

        # Take the fourier transform of the data
        fft = torch.rfft(x.data, signal_ndim=self.dim, normalized=False, onesided=False)
        real, imag = fft.split(1, -1)

        # We need to linearize the matrices so we can easily do the multiplication
        real = real.squeeze().view(len(x.size), -1).permute(1, 0)
        imag = imag.squeeze().view(len(x.size), -1).permute(1, 0)

        # Cholskey solve for the inverse operator
        if inverse:
            real_app = self._solve_cholskey(real)
            imag_app = self._solve_cholskey(imag)

        # Otherwise apply the forward operator via matrix multiply
        else:
            real_app = torch.matmul(self.L, real.unsqueeze(-1)).squeeze()
            imag_app = torch.matmul(self.L, imag.unsqueeze(-1)).squeeze()

        # Return tensors to their actual shapes
        real_app = real_app.permute(1, 0).view(len(x.size), *x.size.long())
        imag_app = imag_app.permute(1, 0).view(len(x.size), *x.size.long())
        smooth_fft = torch.stack((real_app, imag_app), -1)

        # Inverse fourier transform
        out_tensor = torch.irfft(smooth_fft, signal_ndim=self.dim, normalized=False, onesided=False)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out

    def apply_forward(self, x):
        return self.forward(x, inverse=False)

    def apply_inverse(self, x):
        return self.forward(x, inverse=True)

    def project_incompressible(self, x):

        fft = torch.rfft(x.data, signal_ndim=self.dim, normalized=False, onesided=False)
        real, imag = fft.split(1, -1)

        # We need to linearize the matrices so we can easily do the multiplication
        real = real.squeeze().view(len(x.size), -1).permute(1, 0)
        imag = imag.squeeze().view(len(x.size), -1).permute(1, 0)

        real_dot = (real * self.sin_grids).sum(-1, keepdim=True)
        imag_dot = (imag * self.sin_grids).sum(-1, keepdim=True)

        real_app = real - (real_dot * self.sin_grids / self.nsq.unsqueeze(-1))
        imag_app = imag - (imag_dot * self.sin_grids / self.nsq.unsqueeze(-1))

        # Deal with the nan
        real_app[real_app != real_app] = real.squeeze()[real_app != real_app]
        imag_app[imag_app != imag_app] = imag.squeeze()[imag_app != imag_app]

        # Return tensors to their actual shapes
        real_app = real_app.permute(1, 0).view(len(x.size), *x.size.long())
        imag_app = imag_app.permute(1, 0).view(len(x.size), *x.size.long())
        smooth_fft = torch.stack((real_app, imag_app), -1)

        # Inverse fourier transform
        out_tensor = torch.irfft(smooth_fft, signal_ndim=self.dim, normalized=False, onesided=False)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
