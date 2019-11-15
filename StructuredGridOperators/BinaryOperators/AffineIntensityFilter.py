import torch
import torch.nn as nn

from CAMP.Core import StructuredGrid
from CAMP.StructuredGridOperators.UnaryOperators.AffineTransformFilter import ApplyGrid
from ._BinaryFilter import Filter


class AffineIntensity(nn.Module):
    def __init__(self, similarity,  dim=2, init_affine=None, device='cpu', dtype=torch.float32):
        super(AffineIntensity, self).__init__()

        if init_affine is None:
            init_affine = torch.eye(dim + 1, dtype=dtype, device=device)[0:dim]

        self.register_buffer('affine', init_affine)

        self.affine.requires_grad = True

        self.device = device
        self.dtype = dtype
        self.dim = dim

        self.similarity = similarity

    @staticmethod
    def Create(similarity,  dim=2, init_affine=None, device='cpu', dtype=torch.float32):
        filt = AffineIntensity(similarity,  dim, init_affine, device, dtype)
        filt = filt.to(device=device, dtype=dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in filt.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return filt

    def apply_affine(self, moving):

        grid = torch.nn.functional.affine_grid(
            self.affine.view(1, 2, 3),
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

        aff_moving = self.apply_affine(moving)

        return self.similarity(target, aff_moving).sum()
