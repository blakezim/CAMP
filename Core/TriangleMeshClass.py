import torch

from CAMP.Core._UnstructuredGridClass import UnstructuredGrid


class TriangleMesh(UnstructuredGrid):
    def __init__(self, vertices, indices, per_vert_values=None, per_index_values=None):
        super(TriangleMesh, self).__init__(vertices, indices, per_vert_values, per_index_values)

        self.normals = None
        self.centers = None

        self.calc_normals()
        self.calc_centers()

        self.flipped_normals = False

    def calc_normals(self):
        tris = self.vertices[self.indices]

        a = tris[:, 0, :]
        b = tris[:, 1, :]
        c = tris[:, 2, :]

        self.normals = 0.5 * torch.cross((a - b), (c - b), dim=1)

    def calc_centers(self, **kwargs):
        tris = self.vertices[self.indices]
        self.centers = (1 / 3.0) * tris.sum(1)

    def flip_normals_(self):
        self.indices = self.indices.flip(-1)
        self.calc_normals()
        self.flipped_normals = True

    def add_surface_(self, verts, indices):
        length_one = len(verts)
        updated_faces = self.indices + length_one
        comb_vertices = torch.cat((verts, self.vertices), 0)
        comb_faces = torch.cat((indices, updated_faces), 0)

        self.vertices = comb_vertices
        self.indices = comb_faces
        self.calc_normals()
        self.calc_centers()

    def __str__(self):
        return f"Triangle Mesh Object - Vertices: {len(self.vertices)}  Faces: {len(self.faces)}"
