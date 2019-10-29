import torch

from Core._UnstructuredGridClass import UnstructuredGrid


class TriangleMesh(UnstructuredGrid):
    def __init__(self, vertices, indices, per_vert_values=None, per_index_values=None):
        super(TriangleMesh, self).__init__(vertices, indices, per_vert_values, per_index_values)

        self.normals = None
        self.centers = None

    def calc_normals(self):
        tris = self.vertices[self.indices]

        a = tris[:, 0, :]
        b = tris[:, 1, :]
        c = tris[:, 2, :]

        self.normals = 0.5 * torch.cross((a - b), (c - b), dim=1)

    def calc_centers(self, **kwargs):
        tris = self.vertices[self.indices]
        self.centers = (1 / 3.0) * tris.sum(1)
