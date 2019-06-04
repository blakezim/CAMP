import sys
import torch
import torch.nn.functional as F

from .GridClass import Grid


class Field:

    def __init__(self, info, grid=None, requires_grad=False,
                 device='cpu', dtype=torch.float32, ftype='HField', space='real'):
        super(Field, self).__init__()

        self.field_type_str = ftype
        self.space = space

        if type(info).__name__ in ['Size', 'list', 'tuple']:

            if grid is None:
                self.grid = Grid(info, device=device, dtype=dtype, requires_grad=requires_grad)
            else:
                self.grid = grid.copy()

            # self.set_identity_(self.grid.size)

        elif type(info).__name__ == 'Grid':
            self.grid = info.copy()
            # self.set_identity_(info.size)

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
        self.set_identity_(self.size)
        self.to(self.device)
        self.to_type_(self.type)

        if len(self.t.shape) == 4:
            self.am_3d = True
        else:
            self.am_3d = False

    def _get_identity(self, shape):
        """Assuming z, x, y"""
        vecs = [torch.linspace(-1, 1, shape[x]) for x in range(0, len(shape))]
        grids = torch.meshgrid(vecs)
        grids = [grid.unsqueeze(-1) for grid in grids]

        # Need to flip the x and y dimension as per affine grid
        grids[-2], grids[-1] = grids[-1], grids[-2]

        field = torch.cat(grids, -1)
        field = field.to(self.device)

        if self.space == 'real':
            field = (((field + 1) * (self.size.float() / 2)) * self.spacing) + self.origin

        return field

    def set_identity_(self, shape):
        self.t = self._get_identity(shape)

    def set_spacing_(self, spacing):
        self.grid.set_spacing(spacing)
        self.spacing = self.grid.spacing

    def set_origin_(self, origin):
        self.grid.set_origin(origin)
        self.origin = self.grid.origin

    def set_size_(self, size):

        if self.am_3d:
            mode = 'trilinear'
        else:
            mode = 'bilinear'

        if self.am_3d:
            self.t = self.t.permute(-1, 0, 1, 2)
            self.t = self.t.unsqueeze(0)
            self.t = F.interpolate(self.t, size=size, mode=mode, align_corners=True).squeeze()
            self.t = self.t.permute(1, 2, 3, 0)
        else:
            self.t = self.t.permute(-1, 0, 1)
            self.t = self.t.unsqueeze(0)
            self.t = F.interpolate(self.t, size=size, mode=mode, align_corners=True).squeeze()
            self.t = self.t.permute(1, 2, 0)

        new_spacing = (self.size / torch.tensor(self.t.size()[:-1], device=self.t.device).float()) * self.spacing
        self.set_spacing_(new_spacing)
        self.grid.set_size(size)
        self.size = self.grid.size

    def field_type(self):
        return self.field_type_str

    def to_v_field_(self):

        if self.field_type_str == 'VField':
            print('This field is already a vector field')
        else:
            identity = self._get_identity(self.t.shape)
            self.t -= identity
            self.field_type_str = 'VField'

    def to_h_field_(self):

        if self.field_type_str == 'HField':
            print('This field is already a lookup field')
        else:
            identity = self._get_identity(self.t.shape)
            self.t += identity
            self.field_type_str = 'HField'

    def to_real_(self):

        if self.space == 'real':
            print('This field is already in real space')
        else:
            self.t = (((self.t + 1) * (self.size.float() / 2)) * self.spacing) + self.origin
            self.field_type_str = 'real'

    def to_index_(self):

        if self.space == 'index':
            print('This field is already in index space')
        else:
            self.t = (((self.t - self.origin) / self.spacing) / (self.size.float() / 2)) - 1
            self.field_type_str = 'index'

    def to_type_(self, new_type):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.type(new_type))
            else:
                pass

    def to(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ in ['Tensor', 'Grid']:
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device
        return self

    def is_3d(self):
        return self.am_3d

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_field = type(self)(self.t.shape)
        new_field.__dict__.update(self.__dict__)
        return new_field

    def __str__(self):
        return f"Field Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
