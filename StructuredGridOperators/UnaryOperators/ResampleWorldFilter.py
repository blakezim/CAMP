import torch
import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class ResampleWorld(Filter):
    def __init__(self, grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):
        super(ResampleWorld, self).__init__()

        self.device = device
        self.dtype = dtype

        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode
        self.grid = grid

    @staticmethod
    def Create(grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):
        resamp = ResampleWorld(grid, interp_mode, pad_mode, device, dtype)
        resamp = resamp.to(device)
        resamp = resamp.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in resamp.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return resamp

    def _get_field(self, x):

        # Compute the index point in x's space of the reference grid origin
        start = ((self.grid.origin - x.origin) / x.spacing) * 2.0/x.size - 1

        # Now do the same thing for the other end of the volume
        adj = self.grid.origin + (self.grid.size * self.grid.spacing)
        stop = ((adj - x.origin) / x.spacing) * 2.0/x.size - 1
        shape = self.grid.size

        grid_bases = [torch.linspace(start[x], stop[x], int(shape[x])) for x in range(0, len(shape))]
        grids = torch.meshgrid(grid_bases)
        field = torch.stack(grids, 0)
        field = field.to(self.device)
        field = field.type(self.dtype)

        field = field.data.permute(torch.arange(1, len(field.shape)).tolist() + [0])
        field = field.data.view(1, *field.shape)

        return field

    def forward(self, x):

        resample_field = self._get_field(x)

        # Resample is expecting x, y, z. Because we are in torch land, our fields are z, y, x. Need to flip
        resample_field = resample_field.flip(-1)

        out_tensor = F.grid_sample(x.data.view(1, *x.data.shape),
                                   resample_field,
                                   mode=self.interpolation_mode,
                                   align_corners=True,
                                   padding_mode=self.padding_mode).squeeze(0)

        out = StructuredGrid.FromGrid(
            self.grid,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
