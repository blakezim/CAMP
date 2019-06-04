import math
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(nn.Module):
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

    def __init__(self, channels, kernel_size, sigma, device='cpu', dtype=torch.float32, dim=2):
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
            [torch.arange(size, dtype=torch.float32, device=device) for size in kernel_size]
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
        self.to(device)
        self.type(dtype)

    def to_(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ in ['Tensor', 'Grid']:
                self.__setattr__(attr, val.to(device))
            else:
                pass

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        if type(x).__name__ == 'Image':
            # Clone the input
            out = x.t.clone()
            if x.color:
                out = out.view(1, *out.shape)
            else:
                out = out.view(1, 1, *out.shape)
            out = self.conv(out, weight=self.weight, groups=self.groups, padding=self.padding).squeeze()

        elif type(x).__name__ == 'Field':
            # Clone the input
            out = x.t.clone()
            if x.is_3d():
                out = out.permute(-1, 0, 1, 2)
                out = out.view(1, *out.shape)
                out = self.conv(out, weight=self.weight, groups=self.groups, padding=self.padding).squeeze()
                out = out.permute(1, 2, 3, 0)
            else:
                out = out.permute(-1, 0, 1)
                out = out.view(1, *out.shape)
                out = self.conv(out, weight=self.weight, groups=self.groups, padding=self.padding).squeeze()
                out = out.permute(1, 2, 0)

        return type(x)(out, x.grid)

