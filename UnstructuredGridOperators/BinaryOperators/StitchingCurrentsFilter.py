import torch
import torch.nn as nn

from CAMP.UnstructuredGridOperators.BinaryOperators.CurrentsEnergyFilter import CurrentsEnergy


class StitchingCurrents(nn.Module):

    def __init__(self, src_surface, tar_surface, reference_surface,
                 sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        super(StitchingCurrents, self).__init__()

        self.device = device
        self.dtype = dtype

        self.register_buffer('orig_tar_vertices', tar_surface.vertices.clone())
        self.register_buffer('orig_src_vertices', src_surface.vertices.clone())

        self.register_buffer('tar_vertices', tar_surface.vertices.clone())
        self.register_buffer('tar_indices', tar_surface.indices)
        self.register_buffer('src_vertices', src_surface.vertices.clone())
        self.register_buffer('src_indices', src_surface.indices)
        self.register_buffer('ref_normals', reference_surface.normals.clone())
        self.register_buffer('ref_centers', reference_surface.centers.clone())
        self.sigma = sigma

        # tris = self.src_vertices[self.src_indices]
        #
        # a = tris[:, 0, :]
        # b = tris[:, 1, :]
        # c = tris[:, 2, :]
        #
        # start_src_area = 0.5 * torch.det(torch.stack([a, b, c], dim=2)).clone()
        # self.register_buffer('start_src_area', start_src_area)
        #
        # tris = self.tar_vertices[self.tar_indices]
        #
        # a = tris[:, 0, :]
        # b = tris[:, 1, :]
        # c = tris[:, 2, :]
        #
        # start_tar_area = 0.5 * torch.det(torch.stack([a, b, c], dim=2)).clone()
        # self.register_buffer('start_tar_area', start_tar_area)

        # self.register_buffer('orig_tar_vertices', tar_surface.vertices.clone())

        self.src_vertices.requires_grad = True
        self.tar_vertices.requires_grad = True

        if kernel == 'cauchy':
            self.kernel = self.cauchy

        elif kernel == 'gaussian':
            self.kernel = self.gaussian

    @staticmethod
    def Create(src_surface, tar_surface, ref_surface, sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        deformable = StitchingCurrents(src_surface, tar_surface, ref_surface, sigma, kernel, device, dtype)
        deformable = deformable.to(device=device, dtype=dtype)

        return deformable

    def forward(self):

        tris = self.src_vertices[self.src_indices]

        a = tris[:, 0, :]
        b = tris[:, 1, :]
        c = tris[:, 2, :]

        src_normals = 0.5 * torch.cross((a - b), (c - b), dim=1)
        src_centers = (1 / 3.0) * tris.sum(1)
        # src_area = 0.5 * torch.det(torch.stack([a, b, c], dim=2))

        tris = self.tar_vertices[self.tar_indices]

        a = tris[:, 0, :]
        b = tris[:, 1, :]
        c = tris[:, 2, :]

        tar_normals = 0.5 * torch.cross((a - b), (c - b), dim=1)
        tar_centers = (1 / 3.0) * tris.sum(1)
        # tar_area = 0.5 * torch.det(torch.stack([a, b, c], dim=2))

        # Calculate the energy
        energy = self.energy(src_normals, src_centers, self.ref_normals, self.ref_centers)
        energy += self.energy(tar_normals, tar_centers, self.ref_normals, self.ref_centers)
        energy += 0.8 * self.energy(src_normals, src_centers, tar_normals, tar_centers)
        # energy += 0.1 * ((self.tar_vertices - self.orig_tar_vertices) ** 2).sum()
        # energy += 0.1 * ((self.src_vertices - self.orig_src_vertices) ** 2).sum()
        # energy += 2 * ((self.orig_tar_vertices - self.tar_vertices) ** 2).sum()
        #
        # # # Also make sure the area of the triangles is preserved
        # # orig_tris = orig_surface.vertices[orig_surface.indices]
        # #
        # oa = orig_tris[:, 0, :]
        # ob = orig_tris[:, 1, :]
        # oc = orig_tris[:, 2, :]
        #
        # cur_area = 0.5 * torch.abs(torch.det(torch.stack([a, b, c], dim=2)))
        # org_area = 0.5 * torch.abs(torch.det(torch.stack([oa, ob, oc], dim=2)))
        #
        # energy += torch.abs(org_area - cur_area).sum()

        return energy

    def energy(self, src_normals, src_centers, tar_normals, tar_centers):

        # Calculate the self term
        e1 = torch.mul(torch.mm(src_normals, src_normals.permute(1, 0)),
                       self.kernel(self.distance(src_centers, src_centers), self.sigma)).sum()

        # Calculate the cross term
        e2 = torch.mul(torch.mm(tar_normals, src_normals.permute(1, 0)),
                       self.kernel(self.distance(src_centers, tar_centers), self.sigma)).sum()

        e3 = torch.mul(torch.mm(tar_normals, tar_normals.permute(1, 0)),
                       self.kernel(self.distance(tar_centers, tar_centers), self.sigma)).sum()

        return e1 - 2 * e2 + e3

    @staticmethod
    def distance(src_centers, tar_centers):
        return ((src_centers.permute(1, 0).unsqueeze(0) - tar_centers.unsqueeze(2)) ** 2).sum(1)

    @staticmethod
    def gaussian(d, sigma):
        return (d / (-2 * (sigma ** 2))).exp()

    @staticmethod
    def cauchy(d, sigma):
        return 1 / (1 + (d / sigma)) ** 2
