import torch
import torch.nn.functional as F

from .GridClass import Grid


class Image(Grid):
    def __init__(self, size, spacing=None, origin=None, device='cpu', dtype=torch.float32, requires_grad=False,
                 tensor=None, channels=1):
        super(Image, self).__init__(size, spacing, origin, device=device, dtype=dtype, requires_grad=requires_grad)

        # Should we assume that if the data doesn't have the 1 channel that it was meant to be 1 channel?

        if tensor is None:
            self.data = torch.zeros([channels] + self.size.int().tolist(),
                                    dtype=dtype,
                                    requires_grad=requires_grad,
                                    device=device)
        else:
            if list(tensor.size()) != [channels] + self.size.int().tolist():
                raise RuntimeError(
                    'Tensor.size() and [channels] + size do not match:'
                    f' Tensor Size: {list(tensor.size())[1:]}, Grid Size: {self.size.int().tolist()}\n'
                    'Tensor shape must be [Channels,  optionally (Z), X, Y] '
                )
            self.data = tensor.clone()

    @staticmethod
    def FromGrid(grid, tensor=None, channels=1):
        return Image(grid.size, grid.spacing, grid.origin, grid.device,
                     grid.dtype, grid.requires_grad, tensor, channels)

    # Should I make this inplace?
    def set_size(self, size, inplace=True):
        old_size = self.size
        super(Image, self).set_size(size)

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
            return Image(
                size=size,
                spacing=new_spacing,
                origin=self.origin,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.dtype,
                tensor=out_data,
                channels=out_data.shape[0]
            )

    def to_(self, device):
        super(Image, self).to_(device)
        self.data = self.data.to(device)

    def to_type_(self, new_type):
        super(Image, self).to_type_(new_type)
        self.data = self.data.to(new_type)

    def shape(self):
        return self.data.shape

    def sum(self):
        return self.data.sum()

    def clone(self):
        return self.__copy__()

    def __add__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data + other.data, (self.data + other.data).shape[0])
        else:
            return self.FromGrid(self, self.data + other, (self.data + other).shape[0])

    def __radd__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data + self.data, (other.data + self.data).shape[0])
        else:
            return self.FromGrid(self, other + self.data, (other + self.data).shape[0])

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data - other.data, (self.data - other.data).shape[0])
        else:
            return self.FromGrid(self, self.data - other, (self.data - other).shape[0])

    def __rsub__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data - self.data, (other.data - self.data).shape[0])
        else:
            return self.FromGrid(self, other - self.data, (other - self.data).shape[0])

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data * other.data, (self.data * other.data).shape[0])
        else:
            return self.FromGrid(self, self.data * other, (self.data * other).shape[0])

    def __rmul__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data * self.data, (other.data * self.data).shape[0])
        else:
            return self.FromGrid(self, other * self.data, (other * self.data).shape[0])

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data / other.data, (self.data / other.data).shape[0])
        else:
            return self.FromGrid(self, self.data / other, (self.data / other).shape[0])

    def __rtruediv__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data / self.data, (other.data / self.data).shape[0])
        else:
            return self.FromGrid(self, other / self.data, (other / self.data).shape[0])

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data // other.data, (self.data // other.data).shape[0])
        else:
            return self.FromGrid(self, self.data // other, (self.data // other).shape[0])

    def __rfloordiv__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data // self.data, (other.data // self.data).shape[0])
        else:
            return self.FromGrid(self, other // self.data, (other // self.data).shape[0])

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __pow__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, self.data ** other.data, (self.data ** other.data).shape[0])
        else:
            return self.FromGrid(self, self.data ** other, (self.data ** other).shape[0])

    def __rpow__(self, other):
        if type(other).__name__ in ['Image', 'Field']:
            return self.FromGrid(self, other.data ** self.data, (other.data ** self.data).shape[0])
        else:
            return self.FromGrid(self, other ** self.data, (other ** self.data).shape[0])

    def __ipow__(self, other):
        return self.__pow__(other)

    def __copy__(self):
        return self.FromGrid(self, self.data, self.data.shape[0])

    def __str__(self):
        return f"Image Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
