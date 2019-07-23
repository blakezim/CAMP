import torch
import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class ApplyField(Filter):
    def __init__(self, field, interp_mode='linear', pad_mode='border', apply_space='real'):
        super(ApplyField, self).__init__()

        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode
        self.apply_space = apply_space

        # Add the field to the register_buffer
        self.field = field

    @staticmethod
    def Create(h_field, interp_mode='linear', pad_mode='border', apply_space='real', device='cpu', dtype=torch.float32):
        app = ApplyField(h_field, interp_mode, pad_mode, apply_space)
        app = app.to(device)
        app = app.type(dtype)

        # Can't add Field and Images to the register buffer, so we need to make sure they are on the right device
        for attr, val in app.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
            else:
                pass

        return app

    def to_input_index(self, x):
        field = self.field.clone()

        field = field - x.origin.view(*x.size.shape, *([1] * len(x.size)))
        field = field / (x.spacing * (x.size / 2)).view(*x.size.shape, *([1] * len(x.size)))
        field = field - 1

        field = field.data.permute(torch.arange(1, len(field.shape())).tolist() + [0])
        field = field.data.view(1, *field.shape)

        return field

    def forward(self, x):

        # Define the interpolation mode
        if self.interpolation_mode == 'linear':
            if len(x.shape()[1:]) == 3:
                interp_mode = 'trilinear'
            else:
                interp_mode = 'bilinear'
        else:
            interp_mode = self.interpolation_mode

        out = x.clone()  # Make sure we don't mess with the original tensor

        # Account for if the input is a tensor (unlikely to happen)
        if type(x).__name__ == 'Tensor':
            out = StructuredGrid(
                size=x.shape[1:],
                spacing=self.field.spacing,
                origin=self.field.origin,
                device=self.field.device,
                dtype=self.field.dtype,
                requires_grad=self.field.requires_grad,
                tensor=x.clone(),
                channels=x.shape[0]
            )

        if self.field.data.shape[0] != len(out.size):
            raise RuntimeError(
                f'Can not apply lut with dimension {self.field.data.shape[0]} to data with dimension {len(out.size)}'
            )

        # Make the grid have the index values of the input
        resample_field = self.to_input_index(x)

        # Resample is expecting x, y, z. Because we are in torch land, our fields are z, y, x. Need to flip
        resample_field = resample_field.flip(-1)

        out.data = F.grid_sample(out.data.view(1, *out.data.shape),
                                 resample_field,
                                 mode=interp_mode,
                                 padding_mode=self.padding_mode).squeeze(0)

        return out
