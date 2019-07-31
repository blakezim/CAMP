import math
import torch
import numbers
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
        self.padding += [x // 2 for x in kernel_size]
        self.padding = tuple(self.padding)

        # The gaussian kernel is the product of the gaussian function of each dimension.
        # Because these are small, don't care about doing meshgrid
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
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    @staticmethod
    def Create(channels, kernel_size, sigma, dim=2, device='cpu', dtype=torch.float32):
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

        out_tensor = self.conv(
                x.data.view(1, *x.data.shape),
                weight=self.weight,
                groups=self.groups,
                padding=self.padding
            ).squeeze(0)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
