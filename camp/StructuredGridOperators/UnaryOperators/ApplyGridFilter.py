import torch
import torch.nn.functional as F

from ...Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class ApplyGrid(Filter):
    def __init__(self, grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):
        """
        Base constructor method. Use ApplyGrid.Create to construct an ApplyGrid filter. The base constructor does not allow class:'StructuredGrid' types to be added to the specific memory location, so not all attributes will be on the same memory device.
        """
        super(ApplyGrid, self).__init__()

        self.device = device
        self.dtype = dtype
        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode
        self.grid = grid

    @staticmethod
    def Create(grid, interp_mode='bilinear', pad_mode='zeros', device='cpu', dtype=torch.float32):

        """
        Returns an Apply Grid Filter that contained a deformation field that can be applied to type :class:`~Core.StructuredGrid` and adds all attributes to the appropriate memory device.

        :param grid: The deformation field to be applied by the Apply Grid Filter. This is assumed to be in real-world
            coordinates relative to the spacing and origin of the grid.
        :type grid: :class:`~Core.StructuredGrid`

        :param interp_mode: Resampling interpolation mode to be used when applying the defromation - one of 'bilinear'
            or 'nearest'.  Default: 'bilinear'
        :type interp_mode: str

        :param pad_mode: padding mode for outside grid values - one of 'zeros', 'border', or 'reflection'.
            Default: 'zeros'
        :type pad_mode: str

        :param device: Memory location for the created Apply Grid Filter - one of 'cpu', 'cuda', or 'cuda:X' where X
            specifies the device identifier. Default: 'cpu'
        :type device: str

        :param dtype: Data type for the Apply Grid Filter attributes. Specified from torch memory types. Default:
            'torch.float32'
        :type dtype: str

        .. note:: When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be
            trilinear. However, when the input is 4-D, the interpolation mode will legitimately be bilinear.

        :return: Apply Grid Filter with the specified parameters.

        """

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

    def _to_input_index(self, x):
        """
        Change the attribute grid from real coordinates to index coordinates that can be used with torch.functional.grid_sample.

        :param x: Grid to be changed from real coordinates to index coordinates.
        :type x: :class:`~Core.StructuredGrid`
        :return: Returns a structured grid that applies in torch index-coordinates.

        """
        grid = self.grid.clone()

        # Change the field to be in index space
        grid = grid - x.origin.view(*x.size.shape, *([1] * len(x.size)))
        grid = grid / (x.spacing * (x.size / 2)).view(*x.size.shape, *([1] * len(x.size)))
        grid = grid - 1

        grid = grid.data.permute(torch.arange(1, len(grid.shape())).tolist() + [0])
        grid = grid.data.view(1, *grid.shape)

        return grid

    def forward(self, in_grid, out_grid=None):
        """
        Apply the grid attribute to in_grid.

        :param in_grid: The :class:'StructuredGrid' to apply the grid attribute to.
        :type in_grid: :class:`~Core.StructuredGrid`
        :param out_grid: An optional additional grid that specifies the output grid. If not specified, the output grid will be the same as the input grid.
        :type out_grid: :class:`~Core.StructuredGrid`, optional
        :return: Returns in_grid resampled through the grid attribute onto the out_grid.

        """

        if out_grid is not None:
            x = out_grid.clone()
        else:
            x = in_grid.clone()

        if self.grid.data.shape[0] != len(x.size):
            raise RuntimeError(
                f'Can not apply lut with dimension {self.grid.data.shape[0]} to data with dimension {len(x.size)}'
            )

        # Make the grid have the index values of the input
        resample_grid = self._to_input_index(in_grid)

        # Resample is expecting x, y, z. Because we are in torch land, our fields are z, y, x. Need to flip
        resample_grid = resample_grid.flip(-1)

        out_tensor = F.grid_sample(in_grid.data.view(1, *in_grid.data.shape),
                                   resample_grid,
                                   mode=self.interpolation_mode,
                                   align_corners=True,
                                   padding_mode=self.padding_mode).squeeze(0)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
