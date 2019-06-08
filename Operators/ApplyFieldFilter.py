import torch
import torch.nn.functional as F

from Core.ImageClass import Image
from Core.GridClass import Grid
from ._BaseFilter import Filter


class ApplyField(Filter):
    def __init__(self, h_field, interp_mode='linear', pad_mode='zeros', apply_space='real'):
        super(ApplyField, self).__init__()

        self.interpolation_mode = interp_mode
        self.padding_mode = pad_mode
        self.apply_space = apply_space
        self.field = h_field.data.permute(torch.arange(1, len(h_field.size) + 1), 0)
        self.field = self.field.view(1, *self.field.shape)

    def to_input_index(self, x):
        field = self.field.clone()
        if self.h_field.space == 'index' and self.apply_space == 'real':
            field.to_real_()

        if self.h_field.space == 'real' and self.apply_space == 'index':
            field.to_index_()

        t = (((field - x.origin) / x.spacing) / (x.size / 2)) - 1

        return t.view(1, *t.shape)

    def forward(self, x):

        if self.interpolation_mode == 'linear':
            if len(x.shape[1:]) == 3:
                interp_mode = 'trilinear'
            else:
                interp_mode = 'bilinear'
        else:
            interp_mode = self.interpolation_mode

        out = x.clone()  # Make sure we don't mess with the original tensor

        # Account for if the input is a tensor (unlikely to happen)
        if type(x).__name__ == 'Tensor':
            out = Image.FromGrid(
                Grid(x.shape[1:],
                     device=self.field.device,
                     dtype=self.field.dtype,
                     requires_grad=self.field.requires_grad),
                tensor=x.clone(),
                channels=x.shape[0]
            )

        # Make the grid have the index values of the input
        resample_field = self.to_input_index(x)
        out.data = F.grid_sample(out.data.view(1, *out.data.shape),
                                 resample_field,
                                 mode=interp_mode,
                                 padding_mode=self.padding_mode).squeeze(0)

        return out


#
# def apply_h_real(Input, H_Field, mode=None, align_corners=True):
#     # need to beef up names and consierations for what can be passed into functions
#
#     def to_grid_index():
#         field = H_Field.t.clone()
#         if H_Field.apply_space == 'index':
#             field.to_real_()
#
#         t = (((field - Input.origin) / Input.spacing) / (Input.size.float() / 2)) - 1
#
#         return t.unsqueeze(0)
#
#     if not mode:
#         if Input.is_3d():
#             mode = 'trilinear'
#         else:
#             mode = 'bilinear'
#
#     field = to_grid_index()
#
#     if type(Input).__name__ == 'Image':
#         im = Input.t.clone()  # Make sure we don't mess with the original tensor
#         if Input.is_3d():
#             im = im.unsqueeze(0)
#         else:
#             im = im.unsqueeze(0).unsqueeze(0)
#
#     elif type(Input).__name__ == 'Field':
#         im = Input.t.clone()  # Make sure we don't mess with the original tensor
#         if Input.is_3d():
#             im = im.permute(-1, 0, 1, 2)
#             im = im.unsqueeze(0)
#         else:
#             im = im.permute(-1, 0, 1)
#             im = im.unsqueeze(0)
#     #     if Input.is_3d():
#     #         field = field.permute(-1, 0, 1, 2)
#     #         field = field.unsqueeze(0)
#     #     else:
#     #         field = field.permute(-1, 0, 1)
#     #         field = field.unsqueeze(0)
#
#     resampled = F.grid_sample(im.float(), field.float()).squeeze()
#
#     return Image(resampled)