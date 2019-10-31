import torch
import torch.nn as nn

from UnstructuredGridOperators.BinaryOperators.CurrentsEnergyFilter import CurrentsEnergy


# TODO Add the affine and translation creation to the init function


class AffineCurrents(nn.Module):

    def __init__(self, tar_normals, tar_centers, affine, translation,
                 sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        super(AffineCurrents, self).__init__()

        self.device = device
        self.dtype = dtype

        # self.sigma = sigma
        # self.affine = affine
        # self.translation = translation
        self.currents = CurrentsEnergy.Create(tar_normals, tar_centers, sigma=sigma, kernel=kernel, device=device)
        # self.tar_normals = tar_normals
        # self.tar_centers = tar_centers

        # Need to add everything to the buffer if we want it to switch!
        # This should be fine as all of these are just tensors
        # self.register_buffer('sigma', sigma) # Floats do not need to be switched
        self.register_buffer('affine', affine)
        self.register_buffer('translation', translation)
        self.register_buffer('tar_normals', tar_normals)
        self.register_buffer('tar_centers', tar_centers)

    @staticmethod
    def Create(tar_normals, tar_centers, affine, translation, sigma,
               kernel='cauchy', device='cpu', dtype=torch.float32):
        aff = AffineCurrents(tar_normals, tar_centers, affine, translation, sigma, kernel, device, dtype)
        aff = aff.to(device=device, dtype=dtype)

        return aff

    def forward(self, src_normals, src_centers):

        # Compute the scaling factor for the affine transformation
        ns = torch.det(self.affine) * torch.t(torch.inverse(self.affine))

        # Update the mean
        ut = src_centers.mean(0) + self.translation

        # Scale the source normals
        src_nrm_sc = torch.mm(ns, src_normals.permute(1, 0)).permute(1, 0)

        # Affine transform the source centers about the mean
        src_cnt_tf = torch.mm(self.affine, (src_centers - src_centers.mean(0)).permute(1, 0)).permute(1, 0) + ut

        # Calculate the energy
        energy = self.currents(src_nrm_sc, src_cnt_tf, self.tar_normals, self.tar_centers)

        return energy
