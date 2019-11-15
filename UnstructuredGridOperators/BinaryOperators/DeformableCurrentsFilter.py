import torch
import torch.nn as nn

from CAMP.UnstructuredGridOperators.BinaryOperators.CurrentsEnergyFilter import CurrentsEnergy


class DeformableCurrents(nn.Module):

    def __init__(self, src_surface, tar_surface, sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        super(DeformableCurrents, self).__init__()

        self.device = device
        self.dtype = dtype

        self.register_buffer('tar_normals', tar_surface.normals)
        self.register_buffer('tar_centers', tar_surface.centers)
        self.register_buffer('src_vertices', src_surface.vertices.clone())
        self.register_buffer('src_indices', src_surface.indices)

        self.src_vertices.requires_grad = True

        self.currents = CurrentsEnergy.Create(self.tar_normals, self.tar_centers,
                                              sigma=sigma, kernel=kernel, device=device)

    @staticmethod
    def Create(src_surface, tar_surface, sigma, kernel='cauchy', device='cpu', dtype=torch.float32):
        deformable = DeformableCurrents(src_surface, tar_surface, sigma, kernel, device, dtype)
        deformable = deformable.to(device=device, dtype=dtype)

        return deformable

    # We want to differentiate with respect to the verticies, not the centers.
    # The CurrentsEnergyFilter takes the normals and centers, so here we need to compute them again from the vertices
    # so we can propagate back to the vertices.
    def forward(self):

        tris = self.src_vertices[self.src_indices]

        a = tris[:, 0, :]
        b = tris[:, 1, :]
        c = tris[:, 2, :]

        normals = 0.5 * torch.cross((a - b), (c - b), dim=1)
        centers = (1 / 3.0) * tris.sum(1)

        # Calculate the energy
        energy = self.currents(normals, centers, self.tar_normals, self.tar_centers)

        # # Also make sure the area of the triangles is preserved
        # orig_tris = orig_surface.vertices[orig_surface.indices]
        #
        # oa = orig_tris[:, 0, :]
        # ob = orig_tris[:, 1, :]
        # oc = orig_tris[:, 2, :]
        #
        # cur_area = 0.5 * torch.abs(torch.det(torch.stack([a, b, c], dim=2)))
        # org_area = 0.5 * torch.abs(torch.det(torch.stack([oa, ob, oc], dim=2)))
        #
        # energy += area_weight * torch.abs(org_area - cur_area).sum()

        return energy
