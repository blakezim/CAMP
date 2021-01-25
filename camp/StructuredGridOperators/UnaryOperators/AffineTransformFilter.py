import torch

from ...Core.StructuredGridClass import StructuredGrid
from .ApplyGridFilter import ApplyGrid
from ._UnaryFilter import Filter

#TODO Check this filter to make sure the affine and translation are correct


class AffineTransform(Filter):
    def __init__(self, target_landmarks=None, source_landmarks=None, affine=None, rigid=False,
                 interp_mode='bilinear', device='cpu', dtype=torch.float32):
        super(AffineTransform, self).__init__()

        self.device = device
        self.dtype = dtype
        self.rigid = rigid
        self.interp_mode = interp_mode

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
               interp_mode='bilinear', device='cpu', dtype=torch.float32):
        """
        Returns an Affine Transform Filter that can be applied to type :class:`~Core.StructuredGrid`. This can be
        initiated using a pair of landmarks (target and source) or with a pre-defined affine transformation (affine).
        Either both target and source landmarks must be provided OR a pre-defined affine.

        :param target_landmarks: Target or unmoving landmarks selected in the target space. This tensor should be of
            size Nxdim where N is the number of landmarks and dim is the dimensionality of the
            :class:`~Core.StructuredGrid` the affine will be applied to.
        :type target_landmarks: tensor, optional
        :param source_landmarks: Source or moving landmarks selected in the source space. This tensor should be of
            size Nxdim where N is the number of landmarks and dim is the dimensionality of the
            :class:`~Core.StructuredGrid` the affine will be applied to.
        :type source_landmarks: tensor, optional
        :param affine: Pre-defined affine. This should be of shape (dim + 1)x(dim + 1) where the added dimension
            stores the translation.
        :type affine: tensor, optional
        :param rigid: If the affine should be reduced to rigid transform only. Default is False.
        :type rigid: bool
        :param interp_mode: Resampling interpolation mode to be used when applying the defromation - one of 'bilinear'
            or 'nearest'.  Default: 'bilinear'
        :type interp_mode: str
        :param device: Memory location for the created filter - one of 'cpu', 'cuda', or 'cuda:X' where X
            specifies the device identifier. Default: 'cpu'
        :type device: str
        :param dtype: Data type for the filter attributes. Specified from torch memory types. Default:
            'torch.float32'
        :type dtype: str

        .. note:: When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be
            trilinear. However, when the input is 4-D, the interpolation mode will legitimately be bilinear.

        :return: Affine transform filter object with the specified parameters.
        """
        aff = AffineTransform(target_landmarks, source_landmarks, affine, rigid, interp_mode, device, dtype)
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

    def forward(self, x, out_grid=None, xyz_affine=False):
        """
        Resamples the :class:`Core.StructuredGrid` through the affine attribute onto the same grid or the out_grid if
        out_grid is provided.

        :param x: :class:`StructuredGrid` to be transformed by the affine attribute.
        :type x: :class:`Core.StructuredGrid`
        :param out_grid: An optional additional grid that specifies the output grid. If not specified, the output grid
            will be the same as the input grid (x).
        :type out_grid: :class:`Core.StructuredGrid`, optional
        :param xyz_affine: Is affine xyz ordered instead of zyx?
        :type xyz_affine: bool, optional
        :return: Affine transformed :class:`StructredGrid`
        """

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

        if xyz_affine:
            aff_grid.data = aff_grid.data.flip(0)

        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([self.dim] + list(range(0, self.dim)))

        if xyz_affine:
            aff_grid.data = aff_grid.data.flip(0)

        x_tf = ApplyGrid.Create(aff_grid, device=aff_grid.device,
                                dtype=aff_grid.dtype, interp_mode=self.interp_mode)(x, out_grid=out_grid)
        return x_tf
