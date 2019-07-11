import sys
import torch
import torch.nn.functional as F

from Core.ImageClass import Image
from Core.GridClass import Grid
from ._BaseFilter import Filter


class ResampleWorld(Filter):
    def __init__(self, grid, interp_mode='linear', pad_mode='border'):
        super(ResampleWorld, self).__init__()

        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode

        # Add the grid to the register buffer
        self.register_buffer('grid', grid)

    @staticmethod
    def Create(grid, interp_mode='linear', pad_mode='border', device='cpu', dtype=torch.float32):
        resamp = ResampleWorld(grid, interp_mode, pad_mode)
        resamp = resamp.to(device)
        resamp = resamp.type(dtype)
        return resamp

    def _get_field(self, x):
        # This is how large the step is for the identity field over the Input Image
        grid_step_size = 2.0 / x.size.float()

        # Get the difference of the origins (top left point)
        top_left_difference = x.origin - self.grid.origin

        # Get the difference of the bottom right corner
        bottom_right_input = x.origin + (x.size * x.spacing)
        bottom_right_output = self.grid.origin + (self.grid.size * self.grid.spacing)
        bottom_right_difference = bottom_right_output - bottom_right_input

        # Determine the start and stop points for the linsapce over the ouput image
        start = - 1 + (top_left_difference * grid_step_size)
        stop = 1 + (bottom_right_difference * grid_step_size)
        shape = self.grid.size

        grid_bases = [torch.linspace(start[x], stop[x], int(shape[x])) for x in range(0, len(shape))]
        grids = torch.meshgrid(grid_bases)
        grids = [grid.unsqueeze(-1) for grid in grids]  # So we can concatenate

        # Need to flip the x and y dimension as per affine grid
        grids[-2], grids[-1] = grids[-1], grids[-2]
        field = torch.cat(grids, -1)

        # Need to set the device so they can be resampled
        field = field.unsqueeze(0).to(x.device).type(x.dtype)
        return field

    def forward(self, x):

        if self.interpolation_mode == 'linear':
            if len(x.size[1:]) == 3:
                interp_mode = 'trilinear'
            else:
                interp_mode = 'bilinear'
        else:
            interp_mode = self.interpolation_mode

        out = x.clone()  # Make sure we don't mess with the original tensor

        if type(x).__name__ == 'Tensor':
            out = Image.FromGrid(
                Grid(x.shape[1:],
                     device=self.field.device,
                     dtype=self.field.dtype,
                     requires_grad=self.field.requires_grad),
                tensor=x.clone(),
                channels=x.shape[0]
            )

        resample_field = self._get_field(x)
        out.data = F.grid_sample(out.data.view(1, *out.data.shape),
                                 resample_field,
                                 mode=interp_mode,
                                 padding_mode=self.padding_mode).squeeze(0)

        return out
