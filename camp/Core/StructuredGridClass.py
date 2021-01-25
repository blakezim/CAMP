import torch
import torch.nn.functional as F


class StructuredGrid:
    """ This is the base class for grid structured data such as images, look-up tables (luts), vector fields, etc. This
    class wraps a torch tensor (data attribute) to provide world coordinate system context.

    :param size: Size of the grid. Size is ordered [z],y,x ([] is optional).
    :type size: list, tuple, tensor
    :param spacing: Spacing between the grid elements. Default is isotropic 1.0 spacing.
    :type spacing: list, tuple, tensor, optional
    :param origin: Real world location of the pixel (2D) or voxel (3D) with the minimum location value. The locations
        of the grid elements increase by the spacing in each relative direction from this voxel. Default pleaces the
        center of the grid at the origin.
    :type origin: list, tuple, tensor, optional
    :param device: Memory location - one of 'cpu', 'cuda', or 'cuda:X' where X specifies the device identifier.
        Default: 'cpu'
    :type device: str, optional
    :param dtype: Data type, specified from torch memory types. Default: 'torch.float32'
    :type dtype: str, optional
    :param requires_grad: Track tensor for gradient operations. Default: False
    :type requires_grad: bool, optional
    :param tensor: The underlying tensor for the data attribute. This allows :class:`StructuredGird` to be wrapped
        around alread-exisiting tensors. This tensor must be of size C,[z],y,x where [z],y,x are the same as 'size' and
        C is equal to Channels. If not provided, the data attribute will be initialized to size C,[z],y,x with zeros.
    :type tensor: torch.tensor, optional
    :param channels: Number of channels for the grid. For example, black and white images must have 1 channel
        and RGB images have 3 channels. Channels can be any integer number.
    :type channels: int
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
        """
        Construct a new :class:`StructuredGrid` from a reference :class:`StructuredGrid` (for the size, spacing,
        origin, device, dtype, requires_grad) and a torch tensor.

        :param grid: Reference :class:`StructuredGrid` with the reference attributes.
        :type grid: :class:`StructuredGrid`
        :param tensor: Torch tensor to wrap into the new :class:`StructuredGrid`. Must have size [z],y,x from the
            reference :class:`StructuredGrid` and the number of specific channels.
        :type tensor: tensor
        :param channels: Channels of the input tensor.
        :type channels: int
        :return: New :class:`StructuredGrid` wrapped around the input tensor.
        """
        return StructuredGrid(grid.size, grid.spacing, grid.origin, grid.device,
                              grid.dtype, grid.requires_grad, tensor, channels)

    def set_to_identity_lut_(self):
        """
        Set the tensor to an real world identity look-up table (LUT) using the spacing and origin of the
        :class:`StructuredGrid`. The number of channels will be set to the number of dimensions in size.

        :return: :class:`StructuredGrid` as a real world identity LUT.
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
        """
        Set the size of the :class:`StructuredGrid`. This will update the spacing and origin of the
        :class:`StructuredGrid` to maintain the original real world FOV.

        :param size: New size for the :class:`StructuredGrid` [z],y,x
        :type size: torch.tensor
        :param inplace: Perform the resize operation in place. Default=True.
        :type inplace: bool
        :return: If inplace==True then returns a new :class:`StructuredGrid`.
        """
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
        """
        Set the spacing. Does not change the origin.

        :param spacing:  New spacing.
        :type spacing: list, tuple, tensor,
        :return: None
        """
        if type(spacing).__name__ == 'Tensor':
            self.spacing = spacing.clone()
        else:
            self.spacing = torch.tensor(spacing,
                                        dtype=self.dtype,
                                        requires_grad=self.requires_grad,
                                        device=self.device)

    def set_origin_(self, origin):
        """
        Set the origin. Does not change the spacing.

        :param origin:  New origin.
        :type origin: list, tuple, tensor
        :return: None
        """
        if type(origin).__name__ == 'Tensor':
            self.origin = origin.clone()
        else:
            self.origin = torch.tensor(origin,
                                       dtype=self.dtype,
                                       requires_grad=self.requires_grad,
                                       device=self.device)

    def get_subvol(self, zrng=None, yrng=None, xrng=None):

        """
        Extract a sub volume. The coordiantes for the sub volume are in index coordiantes.
        Updates the origin to maintain the world coordinate system location.

        :param zrng: Tuple or list of 2 values between [0, size[0]]. If no range is provided, the size stays the same.
        :type zrng: list, tuple, optional
        :param yrng: Tuple or list of 2 values between [0, size[1]]. If no range is provided, the size stays the same.
        :type yrng: list, tuple, optional
        :param xrng: Tuple or list of 2 values between [0, size[2]]. If no range is provided, the size stays the same.
        :type xrng: list, tuple, optional
        :return: Sub volume with updated origin.
        """

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
        """
        Extract a slice from a 3D volume. Updates the origin to maintain the world coordinate system location.

        :param index: Slice index to extract.
        :type index: int
        :param dim: Dimension along which to extract the slice.
        :type dim: int
        :return: Extracted slice.
        """

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
        """
        Change the memory device of the :class:`StructuredGrid`.

        :param device: New memory location - one of 'cpu', 'cuda', or 'cuda:X' where X specifies the device identifier.
        :type device: str, optional
        :return: None
        """

        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device

    def to_type_(self, new_type):
        """
        Change the data type of the :class:`StructuredGrid` attributes.

        :param dtype: Data type, specified from torch memory types. Default: 'torch.float32'
        :type dtype: str, optional
        :return: None
        """

        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.type(new_type))
            else:
                pass
        self.dtype = new_type

    def copy(self):
        """
        Create a copy of the :class:`StructuredGrid`.

        :return: Copy of :class:`StructuredGrid`.
        """
        return self.__copy__()

    def clone(self):
        """
        Create a copy of the :class:`StructuredGrid`.

        :return: Copy of :class:`StructuredGrid`.
        """
        return self.__copy__()

    def sum(self):
        """
        Sum of the data attribute.

        :return: data.sum()
        """
        return self.data.sum()

    def min(self):
        """
        Min of the data attribute.

        :return: data.min()
        """
        return self.data.min()

    def max(self):
        """
        Max of the data attribute.

        :return: data.max()
        """
        return self.data.max()

    def minmax(self):
        """
        Min and Max of the data attribute.

        :return: [data.min(), data.max()]
        """
        return torch.tensor([self.data.min(), self.data.max()], device=self.device)

    def shape(self):
        """
        Returns the shape of the data attribute, including the channels.

        :return: data.shape
        """
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
