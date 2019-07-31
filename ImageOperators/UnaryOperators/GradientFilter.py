import torch
import torch.nn.functional as F

from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter


class Gradient(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(Gradient, self).__init__()

        self.device = device
        self.dtype = dtype

        pad_vec = tuple([1] * dim * 2)
        kernel = self._create_filters(dim)
        kernel = kernel.unsqueeze(1)
        self.register_buffer('weight', kernel)

        if dim == 1:
            self.conv = F.conv1d
            self.padding = torch.nn.ReplicationPad1d(pad_vec)
        elif dim == 2:
            self.conv = F.conv2d
            self.padding = torch.nn.ReplicationPad2d(pad_vec)
        elif dim == 3:
            self.conv = F.conv3d
            self.padding = torch.nn.ReplicationPad3d(pad_vec)
        else:
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.'
            )

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        grad = Gradient(dim, device, dtype)
        grad = grad.to(device)
        grad = grad.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in grad.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

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
            return torch.cat((y.unsqueeze(0), x.unsqueeze(0)), 0)

        if dim == 3:
            x = torch.mul(x.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
            y = torch.mul(y.unsqueeze(0), avg.unsqueeze(1).unsqueeze(2))
            z = torch.mul(torch.ger(avg, avg).unsqueeze(0), base.unsqueeze(1).unsqueeze(2))
            return torch.cat((z.unsqueeze(0), y.unsqueeze(0), x.unsqueeze(0)), 0)

    def forward(self, x):

        # Put the channels in the batch dimension
        out_tensor = self.conv(
            self.padding(x.data.view(x.data.shape[0], 1, *x.data.shape[1:])),
            weight=self.weight
        )

        # Combine the first dimensions
        out_tensor = out_tensor.view(out_tensor.shape[0] * out_tensor.shape[1], *out_tensor.shape[2:])

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out
