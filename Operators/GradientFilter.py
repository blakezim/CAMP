import torch
import torch.nn as nn
import torch.nn.functional as F


class Gradient(nn.Module):
    def __init__(self, channels, dim=2):
        super(Gradient, self).__init__()

        self.padding = tuple([1] * dim)
        kernel = self._create_filters(dim)

        # Reshape to depthwise convolutional weight
        kernel = kernel.unsqueeze(1)
        # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # if dim == 2:
        #     x, y = self._create_filters(dim)
        #     self.sobel_filt = nn.Conv2d(1, dim, (3, 3), stride=1, bias=False)
        #     self.sobel_filt.weight[0] = x.float()
        #     self.sobel_filt.weight[1] = y.float()
        #
        # elif dim == 3:
        #     z, x, y = self._create_filters(dim)
        #     self.sobel_filt = nn.Conv3d(1, dim, (3, 3, 3), stride=1, padding=1, bias=False)
        #     self.sobel_filt.requires_grad = requires_grad
        #     self.sobel_filt.weight[0] = z
        #     self.sobel_filt.weight[1] = x.float()
        #     self.sobel_filt.weight[2] = y.float()
        self.register_buffer('weight', kernel)
        self.groups = channels * dim

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
    def Create(channels, dim=2, device='cpu', dtype=torch.float32):
        grad = Gradient(channels, dim)
        grad = grad.to(device)
        grad = grad.type(dtype)
        return grad

    @staticmethod
    def _create_filters(dim):

        base = torch.tensor([1, 0, -1], dtype=torch.float32)

        if dim == 1:
            return base

        avg = torch.tensor([0, 1, 0], dtype=torch.float32)
        x = torch.ger(avg, base)
        y = torch.ger(base, avg)

        if dim == 2:
            return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0)

        if dim == 3:
            x = torch.mul(x.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
            y = torch.mul(y.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
            z = torch.mul(torch.ger(avg, avg).unsqueeze(0), base.unsqueeze(1).unsqueeze(2))
            return torch.cat((z.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0)), 0)

    def forward(self, x):

        if type(x).__name__ in ['Image', 'Field']:
            out = x.clone()
            out.data = self.conv(
                out.data.view(1, *out.data.shape),
                weight=self.weight,
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



# def GetFilters(dim=2, filter='prewitt'):
#
#     base = torch.tensor([1, 0, -1])
#
#     if filter == 'sobel':
#         avg = torch.tensor([0, 1, 0])
#     elif filter == 'scharr':
#         avg = torch.tensor([3, 10, 3])
#     elif filter == 'prewitt':
#         avg = torch.tensor([1, 1, 1])
#
#     x = torch.ger(avg, base)
#     y = torch.ger(base, avg)
#
#     if dim == 2:
#         return x, y
#
#     if dim == 3:
#         x = torch.mul(x.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
#         y = torch.mul(y.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
#         z = torch.mul(torch.ger(avg, avg).unsqueeze(0), base.unsqueeze(1).unsqueeze(2))
#         return z, x, y
