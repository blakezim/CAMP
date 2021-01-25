import torch

from ..UnaryOperators.GradientFilter import Gradient
from ._BinaryFilter import Filter


class L2Similarity(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(L2Similarity, self).__init__()

        self.device = device
        self.dtype = dtype
        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    @staticmethod
    def Create(dim=2, device='cpu', dtype=torch.float32):

        """
        Compare two :class:`StructuredGrid` objects using an L2 similarity metric.

        :param dim: Dimensionality of the :class:`StructuredGrid` to be compared (not including channels).
        :type dim: int
        :param device: Memory location - one of 'cpu', 'cuda', or 'cuda:X' where X specifies the device identifier.
            Default: 'cpu'
        :type device: str
        :param dtype: Data type for the attributes. Specified from torch memory types. Default: 'torch.float32'
        :type dtype: str
        :return: L2 comparision object.
        """

        filt = L2Similarity(dim, device, dtype)
        filt = filt.to(device)
        filt = filt.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in filt.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return filt

    def forward(self, target, moving):

        """
        Compare two :class:`StructuredGrid` with L2 similarity metric. This is often used for registration so the
        variables are labeled as target and moving. This function preserves the dimensionality of the original grids.

        :param target: Structured Grid 1
        :type target: :class:`StructuredGrid`
        :param moving: Structured Grid 2
        :type moving: :class:`StructuredGrid`
        :return: L2 similarity as :class:`StructuredGrid`
        """

        return 0.5 * ((moving - target) ** 2)

    def c1(self, target, moving, grads):
        """
        First derivative of the L2 similarity metric.

        :param target: Structured Grid 1
        :type target: :class:`StructuredGrid`
        :param moving: Structured Grid 2
        :type moving: :class:`StructuredGrid`
        :param grads: Gradients of the moving image.
        :type grads: :class:`StructuredGrid`
        :return:
        """
        # grads = self.gradient_operator(moving)
        return (moving - target) * grads
