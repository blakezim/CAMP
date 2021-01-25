import torch

from ._UnaryFilter import Filter


class AffineTransformSurface(Filter):
    def __init__(self, affine, rigid=False, device='cpu', dtype=torch.float32):
        super(AffineTransformSurface, self).__init__()

        self.device = device
        self.dtype = dtype
        self.rigid = rigid
        self.register_buffer('affine', affine)

    @staticmethod
    def Create(affine, rigid=False, device='cpu', dtype=torch.float32):
        aff = AffineTransformSurface(affine, rigid, device, dtype)
        aff = aff.to(device=device, dtype=dtype)

        return aff

    def forward(self, obj_in):

        out = obj_in.copy()

        dim = out.vertices.shape[-1]

        a = self.affine[0:dim, 0:dim].clone()
        t = self.affine[0:dim, dim].clone()

        out.vertices = torch.mm(
            a,
            out.vertices.permute(1, 0)
        ).permute(1, 0)
        out.vertices += t

        return out
