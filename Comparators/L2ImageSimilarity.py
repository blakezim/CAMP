import torch

from ImageOperators.GradientFilter import Gradient
from ._BaseFilter import Filter


class L2Similarity(Filter):
    def __init__(self, dim=2, device='cpu', dtype=torch.float32):
        super(L2Similarity, self).__init__()

        self.gradient_operator = Gradient.Create(dim=dim, device=device, dtype=dtype)

    def forward(self, target, moving):
        return (target - moving) ** 2

    def c1(self, target, moving):
        grads = self.gradient_operator(moving)
        return ((target - moving) * grads) * 0.5
