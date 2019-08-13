import torch
import itertools
import torch.nn.functional as F

from CAMP.Core.StructuredGridClass import StructuredGrid
from .GradientFilter import Gradient
from ._UnaryFilter import Filter


class JacobianDeterminant(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(JacobianDeterminant, self).__init__()

        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.gradient = Gradient(dim=dim)
        self.pad = F.pad
        self.pad_vec = []
        self.pad_vec += [[1, 1] for _ in range(0, dim)]
        self.pad_vec = tuple(itertools.chain.from_iterable(self.pad_vec))

        self.crop_vec = []
        self.crop_vec += [[-1, -1] for _ in range(0, dim)]
        self.crop_vec = tuple(itertools.chain.from_iterable(self.crop_vec))



    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):
        jacb = JacobianDeterminant(dim, device, dtype)
        jacb = jacb.to(device)
        jacb = jacb.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in jacb.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return jacb

    def forward(self, x):

        # # Have to pad the tensor with the correct values otherwise edges will be huge
        # temp = StructuredGrid(
        #     size=x.size + 2,
        #     spacing=x.spacing,
        #     origin=(x.origin - x.spacing),
        #     device=x.device,
        #     dtype=x.dtype,
        # )
        # temp.set_to_identity_lut_()
        # temp[]
        grads = self.gradient(x)
        # Central difference scale the gradients by the spacing
        # grads = grads / 2.0

        if self.dim == 2:
            det = grads[0] * grads[3] - grads[1] * grads[2]
        else:
            det = grads[0] * (grads[4] * grads[8] - grads[5] * grads[7]) - \
                  grads[1] * (grads[3] * grads[8] - grads[5] * grads[6]) + \
                  grads[2] * (grads[3] * grads[7] - grads[4] * grads[6])

        # Crop the output and then pad with ones
        det = self.pad(self.pad(det, pad=self.crop_vec), pad=self.pad_vec, mode='constant', value=1.0)

        out = StructuredGrid.FromGrid(
            x,
            tensor=det.view(1, *det.shape),
            channels=1
        )

        return out
