import torch
import torch.nn.functional as F

from CAMP.Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter

# TODO Should change this so that it doesn't have the grid as a variable


class ApplyGrid(Filter):
    def __init__(self, grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):
        super(ApplyGrid, self).__init__()

        self.device = device
        self.dtype = dtype
        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode
        self.grid = grid

    @staticmethod
    def Create(grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):
        app = ApplyGrid(grid, interp_mode, pad_mode, device, dtype)
        app = app.to(device)
        app = app.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in app.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return app

    def set_size(self, grid):
        return self.Create(
            grid=grid,
            interp_mode=self.interpolation_mode,
            pad_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype
        )

    def to_input_index(self, x):
        grid = self.grid.clone()

        # Change the field to be in index space
        grid = grid - x.origin.view(*x.size.shape, *([1] * len(x.size)))
        grid = grid / (x.spacing * (x.size / 2)).view(*x.size.shape, *([1] * len(x.size)))
        grid = grid - 1

        grid = grid.data.permute(torch.arange(1, len(grid.shape())).tolist() + [0])
        grid = grid.data.view(1, *grid.shape)

        return grid

    def forward(self, in_grid, out_grid=None):

        if out_grid is not None:
            x = out_grid.clone()
        else:
            x = in_grid.clone()

        if self.grid.data.shape[0] != len(x.size):
            raise RuntimeError(
                f'Can not apply lut with dimension {self.grid.data.shape[0]} to data with dimension {len(x.size)}'
            )

        # Make the grid have the index values of the input
        resample_grid = self.to_input_index(in_grid)

        # Resample is expecting x, y, z. Because we are in torch land, our fields are z, y, x. Need to flip
        resample_grid = resample_grid.flip(-1)

        out_tensor = F.grid_sample(in_grid.data.view(1, *in_grid.data.shape),
                                   resample_grid,
                                   mode=self.interpolation_mode,
                                   padding_mode=self.padding_mode).squeeze(0)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
