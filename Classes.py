import sys
import torch

import matplotlib
matplotlib.use('qt5agg')

import Functions as cf


class Grid:
    def __init__(self, size, spacing=None, origin=None,
                 device='cpu', dtype=torch.float32, requires_grad=False):
        super(Grid, self).__init__()

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        if type(size).__name__ == 'Tensor':
            self.size = size.clone().type(torch.int)
        else:
            self.size = torch.tensor(size, dtype=torch.int, requires_grad=requires_grad, device=device)

        if spacing is None:
            self.spacing = torch.ones(len(size), dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(spacing).__name__ == 'Tensor':
                self.spacing = spacing.clone()
            else:
                self.spacing = torch.tensor(spacing, dtype=dtype, requires_grad=requires_grad, device=device)

        if origin is None:
            origin = [-x / 2.0 for x in size]  # For some reason this doesn't work
            self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(origin).__name__ == 'Tensor':
                self.origin = origin.clone()
            else:
                self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)

    def set_size(self, size):
        if type(size).__name__ == 'Tensor':
            self.size = size.clone()
        else:
            self.size = torch.tensor(size,
                                     dtype=self.dtype,
                                     requires_grad=self.requires_grad,
                                     device=self.device)

    def set_spacing(self, spacing):
        if type(spacing).__name__ == 'Tensor':
            self.spacing = spacing.clone()
        else:
            self.spacing = torch.tensor(spacing,
                                        dtype=self.dtype,
                                        requires_grad=self.requires_grad,
                                        device=self.device)

    def set_origin(self, origin):
        if type(origin).__name__ == 'Tensor':
            self.origin = origin.clone()
        else:
            self.origin = torch.tensor(origin,
                                       dtype=self.dtype,
                                       requires_grad=self.requires_grad,
                                       device=self.device)

    def to(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device
        return self

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_grid = type(self)(self.size)
        new_grid.__dict__.update(self.__dict__)
        return new_grid

    def __str__(self):
        return f"Grid Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"


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
            self.t = self.t
            if grid is None:
                self.grid = Grid(info, device=device, dtype=dtype, requires_grad=requires_grad)
            else:
                self.grid = grid.copy()
        else:
            sys.exit('Unknown info type. Please input type: Size, list, tuple, Grid, or Tensor.')

        self.size = self.grid.size
        self.spacing = self.grid.spacing
        self.origin = self.grid.origin
        self.device = self.grid.device
        self.type = self.grid.dtype
        self.requires_grad = self.grid.requires_grad
        self._update()

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

    def _update(self):
        self.to(self.device)
        self.to_type_(self.type)

    def set_spacing(self, spacing):
        self.grid.set_spacing(spacing)
        self.spacing = self.grid.spacing

    def set_origin(self, origin):
        self.grid.set_origin(origin)
        self.origin = self.grid.origin

    def set_size(self, size):
        new_obj = cf.ResampleToSize(self, size=size)
        self.__dict__.update(new_obj.__dict__)

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
