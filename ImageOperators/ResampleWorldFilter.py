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
        self.grid = grid

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

        grid_bases = [torch.linspace(start[x], stop[x], shape[x]) for x in range(0, len(shape))]
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
            if len(x.shape[1:]) == 3:
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


# def ResampleToSize(Im, size=None, mode=None, scale=None, align_corners=True):
#
#     def _resample(im):
#         if size is not None:
#             out = F.interpolate(im, size=size, mode=mode, align_corners=align_corners).squeeze()
#         if scale is not None:
#             out = F.interpolate(im, scale_factor=scale, mode=mode, align_corners=align_corners).squeeze()
#         return out
#
#     # Make sure that either the size or scale is specified
#     if size is None and scale is None:
#         sys.exit('Please specify either a new size or scale factor')
#
#     if size is not None and scale is not None:
#         sys.exit('Im of both size and scale is ambiguous, please only specify size or scale')
#
#     # Define the interpolation mode if one is not specified
#     if not mode:
#         if Im.is_3d():
#             mode = 'trilinear'
#         else:
#             mode = 'bilinear'
#
#     # Make sure we don't mess with the original tensor
#     im = Im.t.clone()
#
#     # Need to do different pre/post processing for Images vs Fields
#     if type(Im).__name__ == 'Image':
#         if Im.color:
#             im = im.unsqueeze(0)
#         else:
#             im = im.unsqueeze(0).unsqueeze(0)
#         out = _resample(im)
#         # Need to deal with the origin and the spacing now that the image has been resized
#         out_grid = cc.Grid(
#             out.size(),
#             spacing=(Im.size / torch.tensor(out.size(), device=Im.device).float()) * Im.spacing,
#             origin=Im.origin,
#             device=Im.device,
#             dtype=Im.type,
#             requires_grad=Im.requires_grad
#         )
#
#     elif type(Im).__name__ == 'Field':
#         if Im.is_3d():
#             im = im.permute(-1, 0, 1, 2)
#             im = im.unsqueeze(0)
#             out = _resample(im)
#             out = out.permute(1, 2, 3, 0)
#         else:
#             im = im.permute(-1, 0, 1)
#             im = im.unsqueeze(0)
#             out = _resample(im)
#             out = out.permute(1, 2, 0)
#
#         # Need to deal with the origin and the spacing now that the image has been resized
#         out_grid = cc.Grid(
#             out.size()[:-1],
#             spacing=(Im.size / torch.tensor(out.size()[:-1], device=Im.device).float()) * Im.spacing,
#             origin=Im.origin,
#             device=Im.device,
#             dtype=Im.type,
#             requires_grad=Im.requires_grad
#         )
#
#     return type(Im)(out, out_grid)

