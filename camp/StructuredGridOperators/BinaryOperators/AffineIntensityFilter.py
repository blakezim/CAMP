import torch
import torch.nn as nn

# from Core import *
# # from camp.Core import StructuredGrid
# from StructuredGridOperators.UnaryOperators.AffineTransformFilter import ApplyGrid
# from ._BinaryFilter import Filter


class AffineIntensity(nn.Module):
    def __init__(self, similarity,  dim=2, init_affine=None, init_translation=None, device='cpu', dtype=torch.float32):
        super(AffineIntensity, self).__init__()

        if init_affine is None:
            init_affine = torch.eye(dim, dtype=dtype, device=device)

        if init_translation is None:
            init_translation = torch.zeros(dim, dtype=torch.float32, device=device)

        self.register_buffer('affine', init_affine)
        self.register_buffer('translation', init_translation)

        self.affine.requires_grad = True
        self.translation.requires_grad = True

        self.device = device
        self.dtype = dtype
        self.dim = dim

        self.similarity = similarity

    @staticmethod
    def Create(similarity,  dim=2, init_affine=None, init_translation=None, device='cpu', dtype=torch.float32):
        """
        Object for registering two structured grids with an affine transformation. The affine and translation are
        optimized independently. This Affine Intensity filter must be on the same device as the target and moving
        structured grids. The affine and translation attributes have requires_grad=True so they can be added to a
        torch optimizer and updated with autograd functions.

        :param similarity: This is the similiarty filter used to compare the two structured grids to be registered.
            This filter is usually a Binary Operator (ie. L2 Image Similarity)
        :type similarity: :class:`Filter`
        :param dim: Dimensionality of the structured grids to be registered (not including channels).
        :type dim: int
        :param init_affine: Initial affine to apply to the moving structured grid.
        :type init_affine: tensor, optional
        :param init_translation: Initial translation to apply to the moving structured grid.
        :type init_translation: tensor, optional
        :param device: Memory location - one of 'cpu', 'cuda', or 'cuda:X' where X specifies the device identifier.
            Default: 'cpu'
        :type device: str
        :param dtype: Data type for the attributes. Specified from torch memory types. Default: 'torch.float32'
        :type dtype: str
        :return: Affine Intensity Filter Object
        """
        filt = AffineIntensity(similarity,  dim, init_affine, init_translation, device, dtype)
        filt = filt.to(device=device, dtype=dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in filt.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return filt

    def _apply_affine(self, moving):

        affine = torch.cat([self.affine, self.translation[:, None]], 1)

        grid = torch.nn.functional.affine_grid(
            affine.view(1, self.dim, self.dim + 1),
            size=torch.Size([1] + list(moving.size())),
            align_corners=True
        )

        out_im = torch.nn.functional.grid_sample(
            moving.unsqueeze(0),
            grid,
            align_corners=True
        ).squeeze(0)

        return out_im

    def forward(self, target, moving):
        """
        Apply the forward affine operation applied to the moving image and calculate the resulting similarity measure
        between the target and moving images. The gradients on the affine and translation attributes are tracked
        through this forward operation so that the gradient update can be applied to update the affine and translation.
        This function is meant to be used iteratively in the registration process.

        :param target: Target structured grid. Does not get updated or changed.
        :type target: :class:`StructuredGrid`
        :param moving: Moving structured grid. Affine and translation are applied this structured grid before the
            similarity calculation.
        :type moving: :class:`StructuredGrid`
        :return: Energy from the similarity evaluation (usually a single float).
        """

        aff_moving = self._apply_affine(moving)

        return self.similarity(target, aff_moving).sum()
