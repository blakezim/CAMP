import sys
import torch
import torch.nn.functional as F

from .GridClass import Grid


class Image(Grid):
    def __init__(self, size, spacing=None, origin=None, device='cpu', dtype=torch.float32, requires_grad=False,
                 tensor=None, channels=1):
        super(Image, self).__init__(size, spacing, origin, device='cpu', dtype=torch.float32, requires_grad=False)

        if tensor is None:
            self.data = torch.zeros([channels] + list(size),
                                    dtype=dtype,
                                    requires_grad=requires_grad,
                                    device=device)
        else:
            if list(tensor.size()) != [channels] + list(size):
                raise RuntimeError(
                    'Tensor.size() and [channels] + size do not match:'
                    f' Tensor Size: {list(tensor.size())[1:]}, Grid Size: {list(size)}\n'
                    'Tensor shape must be [Channels,  optionally (Z), X, Y] '
                )
            self.data = tensor.clone()

    @staticmethod
    def FromGrid(grid, tensor=None, channels=1):
        return Image(grid.size, grid.spacing, grid.origin, grid.device,
                     grid.dtype, grid.requires_grad, tensor, channels)


        # self.size = self.grid.size
        # self.spacing = self.grid.spacing
        # self.origin = self.grid.origin
        # self.device = self.grid.device
        # self.type = self.grid.dtype
        # self.requires_grad = self.grid.requires_grad
        # self.to_(self.device)
        # self.to_type_(self.type)

        # Need to know if the image is color or not - Could probably clean this up in some way
        # if color:
        #     if len(self.size[1:]) == 3:
        #         self.am_3d = True
        #     else:
        #         self.am_3d = False
        # else:
        #     if len(self.size) == 3:
        #         self.am_3d = True
        #     else:
        #         self.am_3d = False
    #
    # def set_spacing_(self, spacing):
    #     self.grid.set_spacing(spacing)
    #     self.spacing = self.grid.spacing

    # def set_origin_(self, origin):
    #     self.grid.set_origin(origin)
    #     self.origin = self.grid.origin

    def set_size(self, size):
        old_size = self.size
        super(Image, self).set_size(size)

        if self.color:
            self.data= self.t.view(1, *self.t.size())
            # self.data= self.t.unsqueeze(0)
        else:
            self.data= self.t.view(1, 1, *self.t.size())

        if self.am_3d:
            mode = 'trilinear'
        else:
            mode = 'bilinear'

        self.data = F.interpolate(self.t, size=size, mode=mode, align_corners=True).squeeze()

        new_spacing = (self.size / torch.tensor(self.t.size(), device=self.device).float()) * self.spacing
        self.set_spacing_(new_spacing)
        self.grid.set_size(size)
        self.size = self.grid.size

    # def to_(self, device):
    #     for attr, val in self.__dict__.items():
    #         if type(val).__name__ in ['Tensor', 'Grid']:
    #             self.__setattr__(attr, val.to(device))
    #         else:
    #             pass
    #     self.device = device

    def to_type_(self, new_type):
        # SUPER THIS
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.type(new_type))
            else:
                pass
        self.type = new_type

    # def is_color(self):
    #     return self.color

    # def is_3d(self):
    #     return self.am_3d

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_image = type(self)(self.t.shape)
        new_image.__dict__.update(self.__dict__)
        return new_image

    def __str__(self):
        return f"Image Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
