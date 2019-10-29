class UnstructuredGrid:
    def __init__(self, vertices, indices, per_vert_values=None, per_index_values=None):

        self.vertices = vertices
        self.indices = indices
        self.per_vert_value = per_vert_values
        self.per_index_value = per_index_values

    def calc_normals(self, **kwargs):
        return None

    def calc_centers(self, **kwargs):
        return None
