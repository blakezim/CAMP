import sys
import torch
import torch.nn.functional as F

from .GridClass import Grid


class Image:
    def __init__(self, info, grid=None, color=False, device='cpu', dtype=torch.float32, requires_grad=False):
        super(Image, self).__init__()

        self.color = color

        if type(info).__name__ in ['Size', 'list', 'tuple']:
            self.t = torch.zeros(info, dtype=dtype, requires_grad=requires_grad, device=device)
            if grid is None:
                self.grid = Grid(info, device=device, dtype=dtype, requires_grad=requires_grad)
            else:
                self.grid = grid.copy()
        elif type(info).__name__ == 'Grid':
            self.t = torch.zeros(info.size, dtype=dtype, requires_grad=requires_grad, device=device)
            self.grid = info.copy()
        elif type(info).__name__ == 'Tensor':
            self.t = info.clone()
            if grid is None:
                self.grid = Grid(info.size(), device=device, dtype=dtype, requires_grad=requires_grad)
            else:
                self.grid = grid.copy()
        else:
            sys.exit('Unknown info type for type Image. Please input type: Size, list, tuple, Grid, or Tensor.')

        self.size = self.grid.size
        self.spacing = self.grid.spacing
        self.origin = self.grid.origin
        self.device = self.grid.device
        self.type = self.grid.dtype
        self.requires_grad = self.grid.requires_grad
        self.to(self.device)
        self.to_type_(self.type)

        # Need to know if the image is color or not - Could probably clean this up in some way
        if color:
            if len(self.size[1:]) == 3:
                self.am_3d = True
            else:
                self.am_3d = False
        else:
            if len(self.size) == 3:
                self.am_3d = True
            else:
                self.am_3d = False

    def set_spacing_(self, spacing):
        self.grid.set_spacing(spacing)
        self.spacing = self.grid.spacing

    def set_origin_(self, origin):
        self.grid.set_origin(origin)
        self.origin = self.grid.origin

    def set_size_(self, size):

        if self.color:
            self.t = self.t.unsqueeze(0)
        else:
            self.t = self.t.unsqueeze(0).unsqueeze(0)

        if self.am_3d:
            mode = 'trilinear'
        else:
            mode = 'bilinear'

        self.t = F.interpolate(self.t, size=size, mode=mode, align_corners=True).squeeze()

        new_spacing = (self.size / torch.tensor(self.t.size(), device=self.device).float()) * self.spacing
        self.set_spacing_(new_spacing)
        self.grid.set_size(size)
        self.size = self.grid.size

    def to(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ in ['Tensor', 'Grid']:
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device
        return self

    def to_type_(self, new_type):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.type(new_type))
            else:
                pass
        self.type = new_type

    def is_color(self):
        return self.color

    def is_3d(self):
        return self.am_3d

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_image = type(self)(self.t.shape)
        new_image.__dict__.update(self.__dict__)
        return new_image

    def __str__(self):
        return f"Image Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
