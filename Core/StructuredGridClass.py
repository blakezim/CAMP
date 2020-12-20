import torch
import torch.nn.functional as F


class StructuredGrid:
    """This is a conceptual class representation of a simple BLE device
    (GATT Server). It is essentially an extended combination of the
    :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    :param client: A handle to the :class:`simpleble.SimpleBleClient` client
        object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    :param addr: Device MAC address, defaults to None
    :type addr: str, optional
    :param addrType: Device address type - one of ADDR_TYPE_PUBLIC or
        ADDR_TYPE_RANDOM, defaults to ADDR_TYPE_PUBLIC
    :type addrType: str, optional
    :param iface: Bluetooth interface number (0 = /dev/hci0) used for the
        connection, defaults to 0
    :type iface: int, optional
    :param data: A list of tuples (adtype, description, value) containing the
        AD type code, human-readable description and value for all available
        advertising data items, defaults to None
    :type data: list, optional
    :param rssi: Received Signal Strength Indication for the last received
        broadcast from the device. This is an integer value measured in dB,
        where 0 dB is the maximum (theoretical) signal strength, and more
        negative numbers indicate a weaker signal, defaults to 0
    :type rssi: int, optional
    :param connectable: `True` if the device supports connections, and `False`
        otherwise (typically used for advertising ‘beacons’).,
        defaults to `False`
    :type connectable: bool, optional
    :param updateCount: Integer count of the number of advertising packets
        received from the device so far, defaults to 0
    :type updateCount: int, optional
        """

    def __init__(self, size, spacing=None, origin=None,
                 device='cpu', dtype=torch.float32, requires_grad=False,
                 tensor=None, channels=1):
        super(StructuredGrid, self).__init__()

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.channels = channels

        if type(size).__name__ == 'Tensor':
            self.size = size.clone().type(self.dtype).to(self.device)
        else:
            self.size = torch.tensor(size, dtype=dtype, requires_grad=requires_grad, device=device)

        if spacing is None:
            self.spacing = torch.ones(len(size), dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(spacing).__name__ == 'Tensor':
                self.spacing = spacing.clone().type(self.dtype).to(self.device)
            else:
                self.spacing = torch.tensor(spacing, dtype=dtype, requires_grad=requires_grad, device=device)

        if origin is None:
            # Is assuming center origin the best thing to do?
            origin = [(-x * s) / 2.0 for x, s in zip(size, self.spacing)]
            self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(origin).__name__ == 'Tensor':
                self.origin = origin.clone().type(self.dtype).to(self.device)
            else:
                self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)

        if tensor is None:
            self.data = torch.zeros([channels] + self.size.int().tolist(),
                                    dtype=dtype,
                                    requires_grad=requires_grad,
                                    device=device)

        else:
            if list(tensor.size()) != [channels] + self.size.int().tolist():
                raise RuntimeError(
                    '{Tensor.shape} and {[channels] + input_size} do not match:'
                    f' Tensor Size: {list(tensor.shape)}, Grid Size: {[channels] + self.size.int().tolist()}\n'
                    'Tensor shape must be [Channels,  optionally (Z), X, Y] '
                )
            self.data = tensor.clone()
            self.data = self.data.to(self.device)
            self.data = self.data.type(self.dtype)

    @staticmethod
    def FromGrid(grid, tensor=None, channels=1):
        return StructuredGrid(grid.size, grid.spacing, grid.origin, grid.device,
                              grid.dtype, grid.requires_grad, tensor, channels)

    def set_to_identity_lut_(self):
        """
        :input: None
        :return: None
        """
        # Create the vectors to be gridded
        vecs = [torch.linspace(-1, 1, int(self.size[x])) for x in range(0, len(self.size))]
        # Create the grids - NOTE: This returns z, y, x
        grids = torch.meshgrid(vecs)
        field = torch.stack(grids, 0)
        field = field.to(self.device)
        field = field.type(self.dtype)

        # Make the field in real coordinates
        field += 1
        field *= ((self.size / 2) * self.spacing).view(*self.size.shape, *([1] * len(self.size)))
        field += self.origin.view(*self.size.shape, *([1] * len(self.size)))

        self.data = field
        self.channels = len(field)

    def set_size(self, size, inplace=True):

        old_size = self.size.clone()

        if type(size).__name__ == 'Tensor':
            new_size = size.clone().to(self.device).type(self.dtype)
        else:
            new_size = torch.tensor(size,
                                    dtype=self.dtype,
                                    requires_grad=self.requires_grad,
                                    device=self.device)
        if len(self.size) == 3:
            mode = 'trilinear'
        else:
            mode = 'bilinear'

        if inplace:
            self.data = self.data.view(1, *self.data.size())
            self.data = F.interpolate(self.data, size=[int(x) for x in new_size.tolist()],
                                      mode=mode, align_corners=True).squeeze(0)
            new_spacing = (old_size / new_size) * self.spacing
            self.set_spacing_(new_spacing)
            self.size = new_size

        else:
            data = self.data.view(1, *self.data.size()).clone()
            out_data = F.interpolate(data, size=[int(x) for x in new_size.tolist()],
                                     mode=mode, align_corners=True).squeeze(0)
            new_spacing = (old_size / new_size) * self.spacing
            return StructuredGrid(
                size=new_size,
                spacing=new_spacing,
                origin=self.origin,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
                tensor=out_data,
                channels=out_data.shape[0]
            )

    def set_spacing_(self, spacing):
        if type(spacing).__name__ == 'Tensor':
            self.spacing = spacing.clone()
        else:
            self.spacing = torch.tensor(spacing,
                                        dtype=self.dtype,
                                        requires_grad=self.requires_grad,
                                        device=self.device)

    def set_origin_(self, origin):
        if type(origin).__name__ == 'Tensor':
            self.origin = origin.clone()
        else:
            self.origin = torch.tensor(origin,
                                       dtype=self.dtype,
                                       requires_grad=self.requires_grad,
                                       device=self.device)

    def get_subvol(self, zrng=None, yrng=None, xrng=None):

        new_origin = self.origin.clone()

        if len(self.size) == 3:
            if zrng is None:
                zrng = [0, self.size[-3]]
            new_origin[-3] = self.origin[-3] + self.spacing[-3]*zrng[0]

        elif len(self.size) == 2 and zrng is not None:
            raise RuntimeError('Can not extract Z data from 2D image')

        if yrng is None:
            yrng = [0, self.size[-2]]
        new_origin[-2] = self.origin[-2] + self.spacing[-2]*yrng[0]

        if xrng is None:
            xrng = [0, self.size[-1]]
        new_origin[-1] = self.origin[-1] + self.spacing[-1] * yrng[0]

        # Get the tensor
        if len(self.size) == 3:
            data = self.data[:, zrng[0]:zrng[1], yrng[0]:yrng[1], xrng[0]:xrng[1]]
        else:
            data = self.data[:, yrng[0]:yrng[1], xrng[0]:xrng[1]]

        new_grid = StructuredGrid(
            size=data.shape[1:],
            spacing=self.spacing,
            origin=new_origin,
            device=self.device,
            dtype=self.dtype,
            tensor=data,
            channels=self.channels
        )

        return new_grid

    def extract_slice(self, index, dim):
        # Create a tuple of slice types with none
        slices = [slice(None, None, None)] * len(self.shape())
        slices[dim + 1] = slice(index, index + 1, None)
        slices = tuple(slices)

        # Extract the slice from the data
        new_tensor = self.data[slices].squeeze(dim + 1)

        # Calculate the new origin in that dimension, the other two should stay the same
        new_origin = torch.cat((self.origin[0:dim], self.origin[dim + 1:]))
        new_spacing = torch.cat((self.spacing[0:dim], self.spacing[dim + 1:]))
        new_size = torch.cat((self.size[0:dim], self.size[dim + 1:]))

        return StructuredGrid(
            new_size,
            new_spacing,
            new_origin,
            self.device,
            self.dtype,
            self.requires_grad,
            new_tensor,
            self.channels
        )

    def to_(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device

    def to_type_(self, new_type):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.type(new_type))
            else:
                pass
        self.dtype = new_type

    def copy(self):
        return self.__copy__()

    def clone(self):
        return self.__copy__()

    def sum(self):
        return self.data.sum()

    def min(self):
        return self.data.min()

    def max(self):
        return self.data.max()

    def minmax(self):
        return torch.tensor([self.data.min(), self.data.max()], device=self.device)

    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data + other.data, (self.data + other.data).shape[0])
        else:
            return self.FromGrid(self, self.data + other, (self.data + other).shape[0])

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data - other.data, (self.data - other.data).shape[0])
        else:
            return self.FromGrid(self, self.data - other, (self.data - other).shape[0])

    def __rsub__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, other.data - self.data, (other.data - self.data).shape[0])
        else:
            return self.FromGrid(self, other - self.data, (other - self.data).shape[0])

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data * other.data, (self.data * other.data).shape[0])
        else:
            return self.FromGrid(self, self.data * other, (self.data * other).shape[0])

    def __rmul__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, other.data * self.data, (other.data * self.data).shape[0])
        else:
            return self.FromGrid(self, other * self.data, (other * self.data).shape[0])

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data / other.data, (self.data / other.data).shape[0])
        else:
            return self.FromGrid(self, self.data / other, (self.data / other).shape[0])

    def __rtruediv__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, other.data / self.data, (other.data / self.data).shape[0])
        else:
            return self.FromGrid(self, other / self.data, (other / self.data).shape[0])

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data // other.data, (self.data // other.data).shape[0])
        else:
            return self.FromGrid(self, self.data // other, (self.data // other).shape[0])

    def __rfloordiv__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, other.data // self.data, (other.data // self.data).shape[0])
        else:
            return self.FromGrid(self, other // self.data, (other // self.data).shape[0])

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __pow__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, self.data ** other.data, (self.data ** other.data).shape[0])
        else:
            return self.FromGrid(self, self.data ** other, (self.data ** other).shape[0])

    def __rpow__(self, other):
        if type(other).__name__ == 'StructuredGrid':
            return self.FromGrid(self, other.data ** self.data, (other.data ** self.data).shape[0])
        else:
            return self.FromGrid(self, other ** self.data, (other ** self.data).shape[0])

    def __ipow__(self, other):
        return self.__pow__(other)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __copy__(self):
        new_grid = type(self)(self.size)
        new_grid.__dict__.update(self.__dict__.copy())
        return new_grid

    def __str__(self):
        return f"Structured Grid Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
