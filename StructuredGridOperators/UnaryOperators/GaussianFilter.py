import math
import torch
import numbers
import itertools

import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class Gaussian(Filter):
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cpu', dtype=torch.float32):
        super(Gaussian, self).__init__()

        self.device = device
        self.dtype = dtype

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.padding = []
        self.padding += [[(x - 1) // 2, (x - 1) // 2 + (x - 1) % 2] for x in kernel_size]
        self.padding = tuple(itertools.chain.from_iterable(self.padding))

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # I think this sets the weights to the kernel we created
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
            self.padding = torch.nn.ReplicationPad1d(self.padding)
        elif dim == 2:
            self.conv = F.conv2d
            self.padding = torch.nn.ReplicationPad2d(self.padding)
        elif dim == 3:
            self.conv = F.conv3d
            self.padding = torch.nn.ReplicationPad3d(self.padding)
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    @staticmethod
    def Create(channels, kernel_size, sigma, dim=2, device='cpu', dtype=torch.float32):

        """
        Create a filter to gaussian blur a :class:`StructuredGrid` with the specified number of channels.

        :param channels: Number of channels in the :class:`StructuredGrid` to be blurred.
        :type channels: int
        :param kernel_size: Size of the kernel to use.
        :type kernel_size: int
        :param sigma: Sigma of the gaussian kernel.
        :type sigma: int
        :param dim: Number of dimensions in the :class:`StructuredGrid` the filter will be applied to (not including
            channels).
        :type dim: int
        :param device: Memory location for the created filter - one of 'cpu', 'cuda', or 'cuda:X' where X
            specifies the device identifier. Default: 'cpu'
        :type device: str
        :param dtype: Data type for the filter attributes. Specified from torch memory types. Default:
            'torch.float32'
        :type dtype: str

        :return: Gaussian filter object with the specified parameters.
        """

        gauss = Gaussian(channels, kernel_size, sigma, dim, device, dtype)
        gauss = gauss.to(device)
        gauss = gauss.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in gauss.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return gauss

    def forward(self, x):

        """
        Apply the gaussian filter to the input :class:`StructuredGrid` x.

        :param x: :class:`StructuredGrid` to apply the gaussian to.
        :type x: :class:`StructuredGrid`

        :return: Gaussian filtered :class:`StructuredGrid`.
        """

        out_tensor = self.conv(
                self.padding(x.data.view(x.data.shape[0], 1, *x.data.shape[1:])),
                weight=self.weight,
                groups=self.groups,
            ).squeeze(1)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
