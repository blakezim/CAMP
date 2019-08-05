import torch

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter
from .AffineTransformFilter import AffineTransform
from .ApplyGridFilter import ApplyGrid


class RadialBasis(Filter):
    def __init__(self, target_landmarks, source_landmarks, sigma=0.001, incompressible=False,
                 device='cpu', dtype=torch.float32):
        super(RadialBasis, self).__init__()

        if target_landmarks.shape != source_landmarks.shape:
            raise RuntimeError(
                'Shape of target and source landmarks do not match: '
                f' Target Shape: {target_landmarks.shape}, Source Shape: {target_landmarks.shape}'
            )

        self.device = device
        self.dtype = dtype
        self.sigma = sigma
        self.num_landmarks = len(source_landmarks)
        self.dim = len(source_landmarks[0])

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
        def _sigma(target, source):
            mat = torch.eye(len(target))

            if self.incomp:
                # Compute the laplacian
                diff = source.float() - target.float()
                exponential = (-(self.sigma ** 2) * (diff ** 2).sum()).exp()

                grad = -2 * (self.sigma ** 2) * diff * exponential
                ggt = torch.ger(grad, -2 * (self.sigma ** 2) * diff) + mat * -2 * (self.sigma ** 2) * exponential

                lap = ((-2 * self.sigma ** 2 + 4*(self.sigma ** 4) * (diff ** 2)) * exponential).sum() * mat
                mat = ggt - lap

            else:
                r = -1 * (self.sigma ** 2) * ((source - target) ** 2).sum()
                mat = mat * r.exp()

            return mat

        dim = self.dim
        b = torch.zeros(((self.num_landmarks * self.dim), (self.num_landmarks * self.dim)))

        for i in range(self.num_landmarks):
            for j in range(i, self.num_landmarks):
                b[(i * dim):(i * dim) + dim, (j * dim):(j * dim) + dim] = _sigma(self.target_landmarks[i],
                                                                                 self.target_landmarks[j])
                if i != j:
                    b[(j * dim):(j * dim) + dim, (i * dim):(i * dim) + dim] = b[(i * dim):(i * dim) + dim,
                                                                                (j * dim):(j * dim) + dim]

        c = (self.affine_landmarks - self.target_landmarks).view(-1)
        x = torch.matmul(b.inverse(), c)
        self.params = x.view(self.num_landmarks, dim)

    def _solve_affine(self):

        source_landmarks = torch.cat((self.source_landmarks, torch.ones(len(self.source_landmarks), 1)), -1).t()
        target_landmarks = torch.cat((self.target_landmarks, torch.ones(len(self.target_landmarks), 1)), -1).t()

        # Solve for the transform between the points
        self.affine = torch.matmul(
            torch.matmul(
                target_landmarks, source_landmarks.t()
            ),
            torch.matmul(
                source_landmarks, source_landmarks.t()
            ).inverse()
        )

        if self.incomp:
            u, _, vt = torch.svd(self.affine[0:self.dim, 0:self.dim])
            self.affine[0:self.dim, 0:self.dim] = torch.matmul(u, vt)

        affine_landmarks = torch.matmul(self.affine, source_landmarks)
        self.affine_landmarks = affine_landmarks.t()[:, 0:self.dim].contiguous()

    def _apply_affine(self, x):

        affine = self.affine.inverse()

        a = affine[0:self.dim, 0:self.dim].view([1] * self.dim + [self.dim] * self.dim)
        t = affine[0:self.dim, self.dim]

        x.data = torch.matmul(a.unsqueeze(0).unsqueeze(0), x.data.permute(1, 2, 0).unsqueeze(-1))
        x.data = (x.data.squeeze() + t).permute(2, 0, 1)

        return x

    def forward(self, x):

        rbf_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        rbf_grid.set_to_identity_lut_()

        temp = StructuredGrid.FromGrid(x, channels=self.dim)

        for i in range(self.num_landmarks):
            temp.set_to_identity_lut_()

            point = self.target_landmarks[i].view([self.dim] + self.dim * [1]).float()
            weight = self.params[i].view([self.dim] + self.dim * [1]).float()

            if self.incomp:

                diff = (temp.data - point).permute(list(range(1, self.dim + 1)) + [0])

                # Compute the laplacian
                # diff = source.float() - target.float()
                exponential = (-(self.sigma ** 2) * (diff ** 2).sum(-1)).exp()

                grad = -2 * (self.sigma ** 2) * diff * exponential.unsqueeze(-1)
                grad = torch.matmul(grad.unsqueeze(-1), (-2 * (self.sigma ** 2) * diff).unsqueeze(-2))
                ggt = grad + torch.diag_embed((-2 * (self.sigma ** 2) * exponential).unsqueeze(-1).repeat([1] * self.dim + [2]))

                # ggt = torch.ger(grad, -2 * (self.sigma ** 2) * diff) + mat * -2 * (self.sigma ** 2) * exponential

                lap = ((-2 * self.sigma ** 2 + 4 * (self.sigma ** 4) * (diff ** 2)) * exponential.unsqueeze(-1)).sum(-1)
                lap = torch.diag_embed(lap.unsqueeze(-1).repeat([1] * self.dim + [2]))

                mat = ggt - lap

                # Scale by the weights
                mat = torch.matmul(mat, weight.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1)).squeeze()

                temp.data = mat.permute([-1] + list(range(0, self.dim)))

                rbf_grid.data = rbf_grid.data + temp.data

            else:
                temp.data = -1 * (self.sigma ** 2) * ((temp - point) ** 2).data.sum(0)
                temp.data = torch.exp(temp.data)

                rbf_grid.data = rbf_grid.data + temp.data * weight

        rbf_grid = self._apply_affine(rbf_grid)

        x_rbf = ApplyGrid.Create(rbf_grid, device=x.device, dtype=x.dtype)(x)

        return x_rbf

