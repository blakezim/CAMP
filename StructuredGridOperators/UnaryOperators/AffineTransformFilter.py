import torch

from CAMP.Core.StructuredGridClass import StructuredGrid
from .ApplyGridFilter import ApplyGrid
from ._UnaryFilter import Filter
# TODO Check this filter to make sure the affine and translation are correct


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

        else:
            self.dim = len(affine) - 1
            self.affine = affine

    @staticmethod
    def Create(target_landmarks=None, source_landmarks=None, affine=None, rigid=False,
               device='cpu', dtype=torch.float32):
        aff = AffineTransform(target_landmarks, source_landmarks, affine, rigid, device, dtype)
        aff = aff.to(device)
        aff = aff.type(dtype)

        if affine is not None:
            aff.affine = affine
        else:
            aff._solve_affine()

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in aff.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return aff

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

        if self.rigid:
            u, _, vt = torch.svd(self.affine)
            self.affine = torch.matmul(u, vt.t())

        # Solve for the translation
        self.translation = self.target_landmarks.mean(0) - torch.matmul(self.affine,
                                                                        self.source_landmarks.mean(0).t()).t()

    def forward(self, x, out_grid=None):

        # Create the grid
        if out_grid is not None:
            aff_grid = StructuredGrid.FromGrid(out_grid, channels=self.dim)
        else:
            aff_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        aff_grid.set_to_identity_lut_()

        # Want to bring the grid the other direction
        affine = torch.eye(self.dim + 1, device=self.device, dtype=self.dtype)

        if 'target_landmarks' in self.__dict__:
            affine[0:self.dim, 0:self.dim] = self.affine
            affine[0:self.dim, self.dim] = self.translation
        else:
            affine = self.affine.clone()

        affine = affine.inverse()
        a = affine[0:self.dim, 0:self.dim]
        t = affine[-0:self.dim, self.dim]

        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([self.dim] + list(range(0, self.dim)))

        x_tf = ApplyGrid.Create(aff_grid, device=aff_grid.device, dtype=aff_grid.dtype)(x, out_grid=out_grid)
        return x_tf
