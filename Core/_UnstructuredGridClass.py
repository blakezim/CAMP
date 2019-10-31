import torch


class UnstructuredGrid:
    def __init__(self, vertices, indices, per_vert_values=None, per_index_values=None,
                 device='cpu', dtype=torch.float32):

        self.vertices = vertices
        self.indices = indices
        self.per_vert_value = per_vert_values
        self.per_index_value = per_index_values
        self.device = device
        self.dtype = dtype

    def calc_normals(self, **kwargs):
        return None

    def calc_centers(self, **kwargs):
        return None

    def copy(self):
        return self.__copy__()

    def to_(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device

    def __copy__(self):
        new_obj = type(self)(self.vertices, self.indices,  self.per_vert_value, self.per_index_value)
        new_obj.__dict__.update(self.__dict__)
        return new_obj
