import torch

from CAMP.Core.StructuredGridClass import StructuredGrid
from .ApplyGridFilter import ApplyGrid
from ._UnaryFilter import Filter


class AffineTransform(Filter):
    def __init__(self, target_landmarks=None, source_landmarks=None, affine=None, rigid=False,
                 device='cpu', dtype=torch.float32):
        super(AffineTransform, self).__init__()

        self.device = device
        self.dtype = dtype
        self.rigid = rigid

        if target_landmarks is not None and source_landmarks is not None:
            if target_landmarks.shape != source_landmarks.shape:
                raise RuntimeError(
                    'Shape of target and source landmarks do not match: '
                    f' Target Shape: {target_landmarks.shape}, Source Shape: {source_landmarks.shape}'
                )
            self.source_landmarks = source_landmarks
            self.target_landmarks = target_landmarks
            self.dim = len(self.source_landmarks[0])
            self._solve_affine()

        else:
            self.dim = len(affine) - 1
            self.affine = affine

    @staticmethod
    def Create(target_landmarks=None, source_landmarks=None, affine=None, rigid=False,
               device='cpu', dtype=torch.float32):
        aff = AffineTransform(target_landmarks, source_landmarks, affine, rigid, device, dtype)
        aff = aff.to(device)
        aff = aff.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in aff.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return aff

    def _solve_affine(self):

        source_landmarks = torch.cat((self.source_landmarks, torch.ones(len(self.source_landmarks), 1)), -1).t()
        target_landmarks = torch.cat((self.target_landmarks, torch.ones(len(self.target_landmarks), 1)), -1).t()

        # Solve for the affine transform between the points
        self.affine = torch.matmul(
            torch.matmul(
                target_landmarks, source_landmarks.t()
            ),
            torch.matmul(
                source_landmarks, source_landmarks.t()
            ).inverse()
        )

        if self.rigid:
            u, _, vt = torch.svd(self.affine[0:self.dim, 0:self.dim])
            self.affine[0:self.dim, 0:self.dim] = torch.matmul(u, vt)

    def forward(self, x):

        # Create the grid
        aff_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        aff_grid.set_to_identity_lut_()

        # Want to bring the grid the other direction
        self.affine = self.affine.inverse()

        a = self.affine[0:self.dim, 0:self.dim].view([1]*self.dim + [self.dim] * self.dim)
        t = self.affine[0:self.dim, self.dim]

        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([self.dim] + list(range(0, self.dim)))

        x_tf = ApplyGrid.Create(aff_grid, device=aff_grid.device, dtype=aff_grid.dtype)(x)
        return x_tf
