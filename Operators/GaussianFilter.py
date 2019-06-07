import math
import torch
import numbers
import torch.nn.functional as F

from ._BaseFilter import Filter


class Gaussian(Filter):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussian, self).__init__()

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
    def Create(channels, kernel_size, sigma, device='cpu', dtype=torch.float32, dim=2):
        gauss = Gaussian(channels, kernel_size, sigma, dim)
        gauss = gauss.to(device)
        gauss = gauss.type(dtype)
        return gauss

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        if type(x).__name__ in ['Image', 'Field']:
            out = x.clone()
            out.data = self.conv(
                out.data.view(1, *out.data.shape),
                weight=self.weight,
                groups=self.groups,
                padding=self.padding
            ).squeeze(0)
            return out

        elif type(x).__name__ == 'Tensor':
            out = x.clone()
            out = self.conv(out, weight=self.weight, groups=self.groups, padding=self.padding).squeeze(0)
            return out
        else:
            raise RuntimeError(
                'Data type not understood for Gaussian Filter:'
                f' Received type: {type(x).__name__}.  Must be type: [Image, Field, Tensor]'
            )
