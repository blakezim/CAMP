import torch

# from Core.StructuredGridClass import StructuredGrid
from ._UnaryFilter import Filter
from .GaussianFilter import Gaussian


class VarianceEqualize(Filter):
    """Takes an Image and gives the variance equalized version.

    I_out, Im: PyCA Image3Ds
    sigma: (scalar) gaussian filter parameter
    eps: (scalar) division regularizer

    sigma is the width (in voxels) of the gaussian kernel
    eps is the regularizer

    for a gaussian kernel k, we have

    I_ve = I'/sqrt(k * I'^2)
    where I' = I - k*I

    """
    def __init__(self, kernel_size=11, sigma=2.0, eps=1e-3, device='cpu', dtype=torch.float32):
        super(VarianceEqualize, self).__init__()

        self.device = device
        self.dtype = dtype

        self.sigma = sigma
        self.eps = eps
        self.kernel_size = kernel_size

    @staticmethod
    def Create(kernel_size=11, sigma=2.0, eps=1e-3, device='cpu', dtype=torch.float32):
        grad = VarianceEqualize(kernel_size, sigma, eps, device, dtype)
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

    def forward(self, x):

        gaussian_filter = Gaussian.Create(x.shape()[0], self.kernel_size, self.sigma, len(x.size))

        I_prime = x - gaussian_filter(x)

        denominator = torch.sqrt(gaussian_filter(I_prime ** 2).data)

        out = I_prime / (denominator + self.eps)

        return out
