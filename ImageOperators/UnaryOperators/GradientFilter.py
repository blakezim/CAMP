import torch
import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class Gradient(Filter):
    def __init__(self, dim=2):
        super(Gradient, self).__init__()

        self.padding = tuple([1] * dim)
        kernel = self._create_filters(dim)

        kernel = kernel.unsqueeze(1)

        self.register_buffer('weight', kernel)
        # self.groups = channels * dim

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
    def Create(dim=2, device='cpu', dtype=torch.float32):
        grad = Gradient(dim)
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

        out = x.clone()
        out.data = self.conv(
            out.data.view(1, *out.data.shape),
            weight=self.weight,
            padding=self.padding
        ).squeeze(0)
        return out
