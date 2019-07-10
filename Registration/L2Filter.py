import torch
import torch.nn.functional as F

from ImageOperators.GradientFilter import Gradient
from ._BaseFilter import Filter


class L2(Filter):
    def __init__(self, dim=2):
        super(L2, self).__init__()

        self.gradient_operator = Gradient(dim=dim)

    def forward(self, target, moving):
        return (target - moving) ** 2

    def c1(self, target, moving):
        grads = self.gradient_operator(moving)
        return 0.5 * (target - moving) * grads
