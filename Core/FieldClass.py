import torch
import torch.nn.functional as F

from .GridClass import Grid


class Field(Grid):

    def __init__(self, size, spacing=None, origin=None, device='cpu', dtype=torch.float32, requires_grad=False,
                 tensor=None, ftype='HField', space='real'):
        super(Field, self).__init__(size, spacing, origin, device=device, dtype=dtype, requires_grad=requires_grad)

        self.field_type = ftype
        self.space = space

        if tensor is None:
            self.data = self._get_identity(size)

        else:
            if list(tensor.size()) != [len(size)] + list(size):
                raise RuntimeError(
                    'Tensor.size() and [len(size)] + list(size) do not match:'
                    f' Tensor Size: {list(tensor.size())}, Grid Size: {[len(size)] + list(size)}\n'
                    'Tensor shape must be [(Z,Y,X),  optionally (Z), X, Y] '
                )
            self.data = tensor.clone()

    @staticmethod
    def FromGrid(grid, tensor=None, ftype='HField', space='real'):
        return Field(grid.size, grid.spacing, grid.origin, grid.device,
                     grid.dtype, grid.requires_grad, tensor, ftype, space)

    def _get_identity(self, size):
        """Assuming z, x, y"""
        vecs = [torch.linspace(-1, 1, size[x]) for x in range(0, len(size))]
        grids = torch.meshgrid(vecs)
        grids = [grid.view(1, *grid.shape) for grid in grids]

        # Need to flip the x and y dimension as per affine grid
        grids[-2], grids[-1] = grids[-1], grids[-2]

        field = torch.cat(grids, 0)
        field = field.to(self.device)

        if self.space == 'real':
            field += 1
            field *= ((self.size / 2) * self.spacing).view(*self.size.shape, *([1] * len(self.size)))
            field += self.origin.view(*self.size.shape, *([1] * len(self.size)))
            # This is a fancy way to expand the attributes to be the correct shape

        return field

    def set_identity(self):
        self.data = self._get_identity(self.size)

    def set_size(self, size, inplace=True):
        old_size = self.size
        super(Field, self).set_size(size)

        self.data = self.data.view(1, *self.data.size())

        if len(self.size) == 3:
            mode = 'trilinear'
        else:
            mode = 'bilinear'

        if inplace:
            self.data = F.interpolate(self.data, size=size, mode=mode, align_corners=True).squeeze(0)
            new_spacing = (old_size / self.size[1:]) * self.spacing
            self.set_spacing(new_spacing)

        else:
            out_data = F.interpolate(self.data, size=size, mode=mode, align_corners=True).squeeze(0)
            new_spacing = (old_size / self.size[1:]) * self.spacing
            return Field(
                size=size,
                spacing=new_spacing,
                origin=self.origin,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.dtype,
                tensor=out_data,
                ftype=self.field_type,
                space=self.space
            )

    def to_v_field(self):

        if self.field_type == 'VField':
            print('This field is already a vector field')
        else:
            identity = self._get_identity(self.t.shape)
            self.data -= identity
            self.field_type = 'VField'

    def to_h_field(self):

        if self.field_type == 'HField':
            print('This field is already a lookup field')
        else:
            identity = self._get_identity(self.t.shape)
            self.data += identity
            self.field_type = 'HField'

    def to_real(self):

        if self.space == 'real':
            print('This field is already in real space')
        else:
            self.data += 1
            self.data *= ((self.size / 2) * self.spacing).view(*self.size.shape, *([1] * len(self.size)))
            self.data += self.origin.view(*self.size.shape, *([1] * len(self.size)))

            self.space = 'real'

    def to_index(self):

        if self.space == 'index':
            print('This field is already in index space')
        else:
            self.data -= self.origin.view(*self.size.shape, *([1] * len(self.size)))
            self.data /= (self.spacing * (self.size / 2)).view(*self.size.shape, *([1] * len(self.size)))
            self.data -= 1
            self.space = 'index'

    def to_(self, device):
        super(Field, self).to_(device)
        self.data = self.data.to(device)

    def to_type_(self, new_type):
        super(Field, self).to_type_(new_type)
        self.data = self.data.to(new_type)

    def shape(self):
        return self.data.shape

    def clone(self):
        return self.__copy__()

    def __add__(self, other):
        return self.FromGrid(self, self.data + other.data, self.data.shape[0])

    def __sub__(self, other):
        return self.FromGrid(self, self.data - other.data, self.data.shape[0])

    def __mul__(self, other):
        return self.FromGrid(self, self.data * other.data, self.data.shape[0])

    def __truediv__(self, other):
        return self.FromGrid(self, self.data / other.data, self.data.shape[0])

    def __floordiv__(self, other):
        return self.FromGrid(self, self.data // other.data, self.data.shape[0])

    def __pow__(self, other):
        return self.FromGrid(self, self.data ** other.data, self.data.shape[0])

    def __copy__(self):
        return self.FromGrid(self, self.data, ftype=self.field_type, space=self.space)

    def __str__(self):
        return f"Field Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
