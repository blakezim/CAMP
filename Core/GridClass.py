import torch


class Grid:
    def __init__(self, size, spacing=None, origin=None,
                 device='cpu', dtype=torch.float32, requires_grad=False):
        super(Grid, self).__init__()

        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        if type(size).__name__ == 'Tensor':
            self.size = size.clone().type(torch.int)
        else:
            self.size = torch.tensor(size, dtype=torch.int, requires_grad=requires_grad, device=device)

        if spacing is None:
            self.spacing = torch.ones(len(size), dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(spacing).__name__ == 'Tensor':
                self.spacing = spacing.clone()
            else:
                self.spacing = torch.tensor(spacing, dtype=dtype, requires_grad=requires_grad, device=device)

        if origin is None:
            origin = [-x / 2.0 for x in size]  # For some reason this doesn't work
            self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)
        else:
            if type(origin).__name__ == 'Tensor':
                self.origin = origin.clone()
            else:
                self.origin = torch.tensor(origin, dtype=dtype, requires_grad=requires_grad, device=device)

    def set_size(self, size):
        if type(size).__name__ == 'Tensor':
            self.size = size.clone()
        else:
            self.size = torch.tensor(size,
                                     dtype=self.dtype,
                                     requires_grad=self.requires_grad,
                                     device=self.device)

    def set_spacing(self, spacing):
        if type(spacing).__name__ == 'Tensor':
            self.spacing = spacing.clone()
        else:
            self.spacing = torch.tensor(spacing,
                                        dtype=self.dtype,
                                        requires_grad=self.requires_grad,
                                        device=self.device)

    def set_origin(self, origin):
        if type(origin).__name__ == 'Tensor':
            self.origin = origin.clone()
        else:
            self.origin = torch.tensor(origin,
                                       dtype=self.dtype,
                                       requires_grad=self.requires_grad,
                                       device=self.device)

    def to(self, device):
        for attr, val in self.__dict__.items():
            if type(val).__name__ == 'Tensor':
                self.__setattr__(attr, val.to(device))
            else:
                pass
        self.device = device
        return self

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_grid = type(self)(self.size)
        new_grid.__dict__.update(self.__dict__)
        return new_grid

    def __str__(self):
        return f"Grid Object - Size: {self.size}  Spacing: {self.spacing}  Origin: {self.origin}"
