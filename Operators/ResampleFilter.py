import sys
import torch
import torch.nn.functional as F

from ._BaseFilter import Filter


class Resample(Filter):
    def __init__(self):
        super(Filter, self).__init__()

    def forward(self, x):
        raise NotImplementedError


def ResampleToSize(Im, size=None, mode=None, scale=None, align_corners=True):

    def _resample(im):
        if size is not None:
            out = F.interpolate(im, size=size, mode=mode, align_corners=align_corners).squeeze()
        if scale is not None:
            out = F.interpolate(im, scale_factor=scale, mode=mode, align_corners=align_corners).squeeze()
        return out

    # Make sure that either the size or scale is specified
    if size is None and scale is None:
        sys.exit('Please specify either a new size or scale factor')

    if size is not None and scale is not None:
        sys.exit('Im of both size and scale is ambiguous, please only specify size or scale')

    # Define the interpolation mode if one is not specified
    if not mode:
        if Im.is_3d():
            mode = 'trilinear'
        else:
            mode = 'bilinear'

    # Make sure we don't mess with the original tensor
    im = Im.t.clone()

    # Need to do different pre/post processing for Images vs Fields
    if type(Im).__name__ == 'Image':
        if Im.color:
            im = im.unsqueeze(0)
        else:
            im = im.unsqueeze(0).unsqueeze(0)
        out = _resample(im)
        # Need to deal with the origin and the spacing now that the image has been resized
        out_grid = cc.Grid(
            out.size(),
            spacing=(Im.size / torch.tensor(out.size(), device=Im.device).float()) * Im.spacing,
            origin=Im.origin,
            device=Im.device,
            dtype=Im.type,
            requires_grad=Im.requires_grad
        )

    elif type(Im).__name__ == 'Field':
        if Im.is_3d():
            im = im.permute(-1, 0, 1, 2)
            im = im.unsqueeze(0)
            out = _resample(im)
            out = out.permute(1, 2, 3, 0)
        else:
            im = im.permute(-1, 0, 1)
            im = im.unsqueeze(0)
            out = _resample(im)
            out = out.permute(1, 2, 0)

        # Need to deal with the origin and the spacing now that the image has been resized
        out_grid = cc.Grid(
            out.size()[:-1],
            spacing=(Im.size / torch.tensor(out.size()[:-1], device=Im.device).float()) * Im.spacing,
            origin=Im.origin,
            device=Im.device,
            dtype=Im.type,
            requires_grad=Im.requires_grad
        )

    return type(Im)(out, out_grid)

