import torch
import numbers
from math import pi

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter
from .ApplyGridFilter import ApplyGrid

class RadialBasis(Filter):
    def __init__(self, target_landmarks, source_landmarks, sigma=0.001, incompressible=False,
                 device='cpu', dtype=torch.float32):
        super(RadialBasis, self).__init__()

        if target_landmarks.shape != source_landmarks.shape:
            raise RuntimeError(
                'Shape of target and source landmarks do not match: '
                f' Target Shape: {target_landmarks.shape}, Source Shape: {source_landmarks.shape}'
            )

        self.device = device
        self.dtype = dtype
        self.num_landmarks = len(source_landmarks)
        self.dim = len(source_landmarks[0])

        if isinstance(sigma, numbers.Number):
            self.sigma = torch.as_tensor([sigma] * self.dim)
        else:
            self.sigma = torch.as_tensor(sigma)

        self.sigma = self.sigma.to(device)
        self.sigma = self.sigma.type(dtype)

        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        self.incomp = incompressible

        self._solve_affine()

        self._solve_matrix_spline()

    @staticmethod
    def Create(target_landmarks, source_landmarks, sigma=0.01, incompressible=False,
               device='cpu', dtype=torch.float32):
        rbf = RadialBasis(target_landmarks, source_landmarks, sigma, incompressible, device, dtype)
        rbf = rbf.to(device)
        rbf = rbf.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in rbf.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return rbf

    def _solve_matrix_spline(self):
        def _sigma(x1, x2):
            mat = torch.eye(len(x1), device=self.device)

            diff = x2.float() - x1.float()
            r = torch.sqrt(1 + (self.sigma * (diff ** 2).sum()))
            mat = mat * r
            return mat

        dim = self.dim
        b = torch.zeros(((self.num_landmarks * self.dim), (self.num_landmarks * self.dim)), device=self.device)

        for i in range(self.num_landmarks):
            for j in range(i, self.num_landmarks):
                b[(i * dim):(i * dim) + dim, (j * dim):(j * dim) + dim] = _sigma(self.target_landmarks[i],
                                                                                 self.target_landmarks[j])
                if i != j:
                    b[(j * dim):(j * dim) + dim, (i * dim):(i * dim) + dim] = b[(i * dim):(i * dim) + dim,
                                                                                (j * dim):(j * dim) + dim]

        c = (self.affine_landmarks - self.target_landmarks).view(-1)
        # c = self.affine_landmarks.view(-1)
        x = torch.matmul(b.inverse(), c)
        self.params = x.view(self.num_landmarks, dim)

    def _solve_affine(self):

        source_landmarks_centered = self.source_landmarks - self.source_landmarks.mean(0)
        target_landmarks_centered = self.target_landmarks - self.target_landmarks.mean(0)

        # Solve for the transform between the points
        self.affine = torch.matmul(
            torch.matmul(
                target_landmarks_centered.t(), source_landmarks_centered
            ),
            torch.matmul(
                source_landmarks_centered.t(), source_landmarks_centered
            ).inverse()
        )

        if self.incomp:
            u, _, vt = torch.svd(self.affine)
            self.affine = torch.matmul(u, vt.t())

        # Solve for the translation
        self.translation = self.target_landmarks.mean(0) - torch.matmul(self.affine, self.source_landmarks.mean(0).t()).t()

        self.affine_landmarks = torch.matmul(self.affine, self.source_landmarks.t()).t().contiguous()
        self.affine_landmarks = self.affine_landmarks + self.translation

    def _apply_affine(self, x):

        affine = torch.eye(4, device=self.device, dtype=self.dtype)
        affine[0:3, 0:3] = self.affine
        affine[0:3, 3] = self.translation
        affine = affine.inverse()
        a = affine[0:self.dim, 0:self.dim]
        t = affine[-0:self.dim, self.dim]

        x.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                              x.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        x.data = (x.data.squeeze() + t).permute([-1] + list(range(0, self.dim)))

        return x

    def forward(self, in_grid, out_grid=None):

        if out_grid is not None:
            x = out_grid.clone()
        else:
            x = in_grid.clone()

        rbf_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        rbf_grid.set_to_identity_lut_()
        temp = StructuredGrid.FromGrid(x, channels=self.dim)
        accu = temp.clone() * 0.0

        for i in range(self.num_landmarks):
            temp.set_to_identity_lut_()

            point = self.target_landmarks[i].view([self.dim] + self.dim * [1]).float()
            weight = self.params[i].view([self.dim] + self.dim * [1]).float()

            sigma = self.sigma.view([self.dim] + [1] * self.dim)

            temp.data = torch.sqrt(1 + sigma * ((temp - point) ** 2).data.sum(0))

            accu.data = accu.data + temp.data * weight

        if self.incomp:
            # Expand the size and spacing variable so they are easily usable
            size = x.size.view(*x.size.shape, *([1] * len(x.size)))
            spacing = x.spacing.view(*x.spacing.shape, *([1] * len(x.spacing)))

            # Create the vectors to mesh grid
            vecs = [torch.arange(0, x, device=x.device).float() for x in x.size]

            # Create the grids from the vectors and stack them in the first dimension
            grids = torch.meshgrid(vecs)
            grids = torch.stack(grids, 0)

            # Compute the cosine grids
            # cos_grids = (2 * torch.cos((2.0 * pi * grids) / size) - 2) / (spacing * spacing)
            sin_grids = torch.sin((2 * pi * grids) / size) / spacing

            # Make the spatial dimensions linear
            # cos_grids = cos_grids.view(len(x.size), -1)
            sin_grids = sin_grids.view(len(x.size), -1)
            nsq = (sin_grids ** 2).sum(0)

            sin_grids = sin_grids.permute(1, 0)

            # Take the fourier transform of the data
            fft = torch.rfft(accu.data, signal_ndim=self.dim, normalized=False, onesided=False)
            real, imag = fft.split(1, -1)

            # We need to linearize the matrices so we can easily do the multiplication
            real = real.squeeze().view(len(x.size), -1).permute(1, 0)
            imag = imag.squeeze().view(len(x.size), -1).permute(1, 0)

            real_dot = (real * sin_grids).sum(-1, keepdim=True)
            imag_dot = (imag * sin_grids).sum(-1, keepdim=True)

            real_app = real - (real_dot * sin_grids / nsq.unsqueeze(-1))
            imag_app = imag - (imag_dot * sin_grids / nsq.unsqueeze(-1))

            # Deal with the nan
            real_app[real_app != real_app] = real.squeeze()[real_app != real_app]
            imag_app[imag_app != imag_app] = imag.squeeze()[imag_app != imag_app]

            # Return tensors to their actual shapes
            real_app = real_app.permute(1, 0).view(len(x.size), *x.size.long())
            imag_app = imag_app.permute(1, 0).view(len(x.size), *x.size.long())
            smooth_fft = torch.stack((real_app, imag_app), -1)

            # Inverse fourier transform
            accu.data = torch.irfft(smooth_fft, signal_ndim=self.dim, normalized=False, onesided=False)

            del sin_grids, real_dot, real_app, imag_app, imag_dot
            torch.cuda.empty_cache()

        rbf_grid.data = rbf_grid.data + accu.data

        rbf_grid = self._apply_affine(rbf_grid)

        x_rbf = ApplyGrid.Create(rbf_grid, device=x.device, dtype=x.dtype)(in_grid, out_grid)

        return x_rbf

