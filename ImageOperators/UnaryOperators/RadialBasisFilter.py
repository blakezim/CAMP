import torch
import numbers

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

            if self.incomp:

                diff = x2 - x1

                numerator = ((self.sigma ** 4) * torch.matmul(torch.ones_like(mat) - mat, diff ** 2)) + (self.sigma ** 2)

                denominator = ((self.sigma ** 2) * (diff ** 2).sum() + 1).pow(3.0/2.0)

                double_grad = numerator / denominator * mat

                lap = (numerator / denominator).sum() * mat

                sigma_mat = (self.sigma ** 4).unsqueeze(0).repeat(self.dim, 1)

                off_diag = torch.ger(diff, diff) * (torch.ones_like(mat) - mat) * sigma_mat

                ggt = off_diag + double_grad

                # diff = x2 - x1
                # exponential = (-(self.sigma ** 2) * (diff ** 2).sum()).exp()
                #
                # grad = -2 * (self.sigma ** 2) * diff * exponential
                # ggt = torch.ger(grad, -2 * (self.sigma ** 2) * diff) + mat * -2 * (self.sigma ** 2) * exponential
                #
                # lap = ((-2 * self.sigma ** 2 + 4*(self.sigma ** 4) * (diff ** 2)) * exponential).sum() * mat
                mat = ggt - lap

            else:
                diff = x2.float() - x1.float()
                r = torch.sqrt(1 + (self.sigma * (diff ** 2).sum()))
                mat = mat * r
                # r = -1 * (self.sigma ** 2) * ((x2 - x1) ** 2).sum()
                # mat = mat * r.exp()

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

        source_landmarks = torch.cat((self.source_landmarks,
                                      torch.ones(len(self.source_landmarks), 1, device=self.device)), -1).t()
        target_landmarks = torch.cat((self.target_landmarks,
                                      torch.ones(len(self.target_landmarks), 1, device=self.device)), -1).t()

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

        a = affine[0:self.dim, 0:self.dim].view([1] * self.dim + [self.dim, self.dim])
        t = affine[0:self.dim, self.dim]

        x.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                              x.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        x.data = (x.data.squeeze() + t).permute([-1] + list(range(0, self.dim)))

        return x

    def forward(self, x):

        rbf_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        rbf_grid.set_to_identity_lut_()
        temp = StructuredGrid.FromGrid(x, channels=self.dim)
        temp_zeros = temp.clone() * 0.0

        for i in range(self.num_landmarks):
            temp.set_to_identity_lut_()

            point = self.target_landmarks[i].view([self.dim] + self.dim * [1]).float()
            weight = self.params[i].view([self.dim] + self.dim * [1]).float()

            if self.incomp:

                diff = (temp.data - point).permute(list(range(1, self.dim + 1)) + [0])

                # ones = torch.ones(list(diff.data.shape) + [self.dim], device=self.device, dtype=self.dtype)
                # eyes = torch.diag_embed(torch.ones_like(diff))
                ones = torch.ones([1] * self.dim + [self.dim, self.dim], device=self.device, dtype=self.dtype)
                eyes = torch.eye(self.dim, device=self.device, dtype=self.dtype).view([1] * self.dim + [self.dim, self.dim])
                # off_diag = ones - eyes
                double_grad = (self.sigma ** 4) * torch.matmul(ones - eyes, (diff ** 2).unsqueeze(-1)).squeeze(-1) + (self.sigma ** 2)

                double_grad = double_grad / ((self.sigma ** 2) * (diff ** 2).sum(-1, keepdim=True) + 1).pow(3.0 / 2.0)

                # double_grad = torch.matmul(eyes, double_grad.unsqueeze(-1)).squeeze(-1)

                # del num, denom
                # torch.cuda.empty_cache()

                lap = eyes * double_grad.sum(-1, keepdim=True).unsqueeze(-1)

                ggt = torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2)).squeeze() * (ones - eyes)

                ggt = ggt * self.sigma.unsqueeze(0).repeat(self.dim, 1).view([1]*self.dim + [self.dim, self.dim])

                ggt = ggt + torch.diag_embed(double_grad)

                mat = ggt - lap

                mat = torch.matmul(mat, weight.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1)).squeeze()

                temp.data = mat.permute([-1] + list(range(0, self.dim)))

                temp_zeros.data = temp_zeros.data + temp.data

                rbf_grid.data = rbf_grid.data + temp.data
                #
                # exponential = (-(self.sigma ** 2) * (diff ** 2).sum(-1, keepdim=True)).exp()
                #
                # grad = -2 * (self.sigma ** 2) * diff * exponential #.unsqueeze(-1)
                # grad = torch.matmul(grad.unsqueeze(-1), (-2 * (self.sigma ** 2) * diff).unsqueeze(-2))
                # # ggt = grad + torch.diag_embed((-2 * (self.sigma ** 2) * exponential).unsqueeze(-1).repeat([1] * self.dim + [self.dim]))
                # ggt = grad + torch.diag_embed((-2 * (self.sigma ** 2) * exponential))
                #
                # # lap = ((-2 * self.sigma ** 2 + 4 * (self.sigma ** 4) * (diff ** 2)) * exponential.unsqueeze(-1)).sum(-1)
                # lap = ((-2 * self.sigma ** 2 + 4 * (self.sigma ** 4) * (diff ** 2)) * exponential).sum(-1)
                # lap = torch.diag_embed(lap.unsqueeze(-1).repeat([1] * self.dim + [self.dim]))
                #
                # mat = ggt - lap
                #
                # # Scale by the weights
                # mat = torch.matmul(mat, weight.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1)).squeeze()
                #
                # temp.data = mat.permute([-1] + list(range(0, self.dim)))
                #
                # rbf_grid.data = rbf_grid.data + temp.data

            else:
                sigma = self.sigma.view([self.dim] + [1]*self.dim)
                # temp.data = -1 * (sigma ** 2) * ((temp - point) ** 2).data.sum(0)
                # temp.data = torch.exp(temp.data)

                temp.data = torch.sqrt(1 + sigma * ((temp - point) ** 2).data.sum(0))

                rbf_grid.data = rbf_grid.data + temp.data * weight

        # temp.set_to_identity_lut_()

        # del temp
        # torch.cuda.empty_cache()
        #
        from ImageOperators.UnaryOperators.JacobianDeterminantFilter import JacobianDeterminant
        jacobian = JacobianDeterminant.Create(dim=2, device=self.device, dtype=self.dtype)
        test = jacobian(rbf_grid)

        rbf_grid = self._apply_affine(rbf_grid)

        x_rbf = ApplyGrid.Create(rbf_grid, device=x.device, dtype=x.dtype)(x)

        return x_rbf

