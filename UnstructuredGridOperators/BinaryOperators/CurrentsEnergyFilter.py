import torch
import torch.nn as nn


class CurrentsEnergy(nn.Module):

    def __init__(self, tar_normals, tar_centers, sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        super(CurrentsEnergy, self).__init__()

        if kernel == 'cauchy':
            self.kernel = self.cauchy

        elif kernel == 'gaussian':
            self.kernel = self.gaussian

        else:
            Exception(f'Kernel of type {kernel} is not supported. Please use cauchy or gaussian.')

        self.device = device
        self.dtype = dtype

        self._calc_e3(tar_normals, tar_centers, sigma)
        self.sigma = sigma

        # self.register_buffer('sigma', sigma)

    @staticmethod
    def Create(tar_normals, tar_centers, sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        ce = CurrentsEnergy(tar_normals, tar_centers, sigma, kernel, device, dtype)
        ce = ce.to(device=device, dtype=dtype)

        return ce

    def forward(self, src_normals, src_centers, tar_normals, tar_centers): #, affine, translation, sigma, src_mean):

        # Compute the scaling factor for the affine transformation
        # ns = torch.det(affine) * torch.t(torch.inverse(affine))

        # Update the mean
        # ut = src_mean + translation

        # Scale the source normals
        # src_nrm_sc = torch.mm(ns, src_normals.permute(1, 0)).permute(1, 0)

        # # Affine transform the source centers about the mean
        # src_cnt_tf = torch.mm(affine, (src_centers - src_mean).permute(1, 0)).permute(1, 0) + ut

        # Calculate the self term
        e1 = torch.mul(torch.mm(src_normals, src_normals.permute(1, 0)),
                       self.kernel(self.distance(src_centers, src_centers), self.sigma)).sum()

        # Calculate the cross term
        e2 = torch.mul(torch.mm(tar_normals, src_normals.permute(1, 0)),
                       self.kernel(self.distance(src_centers, tar_centers), self.sigma)).sum()

        return e1 - 2 * e2 + self.e3

    def _calc_e3(self, tar_normals, tar_centers, sigma):
        self.e3 = torch.mul(torch.mm(tar_normals, tar_normals.permute(1, 0)),
                            self.kernel(self.distance(tar_centers, tar_centers), sigma)).sum()

    @staticmethod
    def distance(src_centers, tar_centers):
        return ((src_centers.permute(1, 0).unsqueeze(0) - tar_centers.unsqueeze(2)) ** 2).sum(1)

    @staticmethod
    def colordiff(src_colors, tar_colors):
        return ((src_colors.permute(1, 0).unsqueeze(0) - tar_colors.unsqueeze(2)) ** 2).sum(1)

    @staticmethod
    def gaussian(d, sigma):
        return (d / (-2 * (sigma ** 2))).exp()

    @staticmethod
    def cauchy(d, sigma):
        return 1 / (1 + (d / sigma)) ** 2
