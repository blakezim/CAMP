import sys
import torch
import torch.nn.functional as F

from Core.ImageClass import Image
from Core.GridClass import Grid
from ._UnaryFilter import Filter


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
        field = torch.stack(grids, 0)
        field = field.to(self.device)
        field = field.type(self.dtype)
        field = field.unsqueeze(0)

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

        resample_field = self._get_field(x)
        out.data = F.grid_sample(out.data.view(1, *out.data.shape),
                                 resample_field,
                                 mode=interp_mode,
                                 padding_mode=self.padding_mode).squeeze(0)

        return out
