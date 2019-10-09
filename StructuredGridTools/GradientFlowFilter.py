import torch
import torch.nn.functional as F
from CAMP.StructuredGridOperators import Gradient
from CAMP.Core.StructuredGridClass import StructuredGrid

from ._BaseTool import Filter


# TODO Multiscale matching
# TODO Use only one applier
# TODO Check for memery saving and computation saving areas

class IterativeMatch(Filter):
    def __init__(self, source, target, similarity, operator, regularization=None, step_size=0.001,
                 regularization_weight=0.1, incompressible=True, device='cpu', dtype=torch.float32):
        super(IterativeMatch, self).__init__(source, target)

        if not type(source).__name__ == 'StructuredGrid':
            raise RuntimeError(
                f'Only type "StructuredGrid" for source is accepted, got {type(source).__name__}'
            )

        if not type(target).__name__ == 'StructuredGrid':
            raise RuntimeError(
                f'Only type "StructuredGrid" for target is accepted, got {type(target).__name__}'
            )

        if not any([source.size[x].item() == target.size[x].item() for x in range(len(source.size))]):
            raise RuntimeError(
                f'Images must have the same size - Target Size: {target.size}, Source Size: {source.size}'
            )

        self.device = device
        self.dtype = dtype

        self.similarity = similarity
        self.regularization = regularization
        self.operator = operator
        self.step_size = step_size
        self.reg_weight = regularization_weight
        self.incompressible = incompressible

        self.target = target.clone()
        self.source = source.clone()
        self.moving = source.clone()
        self.field = StructuredGrid.FromGrid(source)
        self.field.set_to_identity_lut_()
        self.update = StructuredGrid.FromGrid(source)
        self.update.set_to_identity_lut_()
        self.initial_energy = self.energy()

        self.gradients = Gradient.Create(dim=len(source.size), device=device, dtype=dtype)(source)
        self.moving_grads = self.gradients.clone()

    @staticmethod
    def Create(source, target, similarity, operator, regularization=None, step_size=0.001,
               regularization_weight=0.1, incompressible=True, device='cpu', dtype=torch.float32):

        match = IterativeMatch(source, target, similarity, operator, regularization, step_size,
                               regularization_weight, incompressible, device, dtype)
        match = match.to(device=device, dtype=dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in match.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        return match

    def energy(self):
        energy = self.similarity(self.target, self.moving).sum()

        if self.regularization:
            reg_e = 0.25 * self.reg_weight * self.regularization(self.field - self.identity, self.operator).sum()

        return energy.item()

    @staticmethod
    def _apply_field(x, field, interpolation_mode='bilinear', padding_mode='zeros'):

        grid = field.clone()

        # Change the field to be in index space
        grid = grid - x.origin.view(*x.size.shape, *([1] * len(x.size)))
        grid = grid / (x.spacing * (x.size / 2)).view(*x.size.shape, *([1] * len(x.size)))
        grid = grid - 1

        grid = grid.data.permute(torch.arange(1, len(grid.shape())).tolist() + [0])
        grid = grid.data.view(1, *grid.shape)

        resample_grid = grid.flip(-1)

        out_tensor = F.grid_sample(x.data.view(1, *x.data.shape),
                                   resample_grid,
                                   mode=interpolation_mode,
                                   padding_mode=padding_mode).squeeze(0)

        out = StructuredGrid.FromGrid(
            x,
            tensor=out_tensor,
            channels=out_tensor.shape[0]
        )

        return out

    def step(self):

        # Calculate the similarity body force
        body_v = self.similarity.c1(self.target, self.moving, self.moving_grads)

        # Apply the operator to the body force
        body_v = self.operator.apply_inverse(body_v)

        # Apply the step size
        body_v = self.step_size*body_v

        if self.incompressible:
            body_v = self.operator.project_incompressible(body_v)

        # Create the update field
        self.update = self.update - body_v

        # Sample the field at the locations of the update field
        self.field = self._apply_field(self.field, self.update, padding_mode="border")

        self.moving = self._apply_field(self.source, self.field)
        self.moving_grads = self._apply_field(self.gradients, self.field)
        # self.update = self.field.clone()
        self.update.set_to_identity_lut_()

        # Calculate and return the new energy
        new_energy = self.energy()
        return new_energy

    def get_field(self):
        return self.field

    def get_image(self):
        return self.moving
