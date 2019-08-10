import torch
from CAMP.ImageOperators import ApplyGrid
from CAMP.Core.StructuredGridClass import StructuredGrid

from ._BaseTool import Filter


class IterativeMatch(Filter):
    def __init__(self, source, target, similarity, operator, regularization=None, step_size=0.001,
                 regularization_weight=0.1, device='cpu', dtype=torch.float32):
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

        self.moving = source.clone()
        self.field = StructuredGrid.FromGrid(source)
        self.field.set_to_identity_lut_()
        self.identity = StructuredGrid.FromGrid(source)
        self.identity.set_to_identity_lut_()
        self.initial_energy = self.energy()

    @staticmethod
    def Create(source, target, similarity, operator, regularization=None, step_size=0.001,
               regularization_weight=0.1, device='cpu', dtype=torch.float32):

        match = IterativeMatch(source, target, similarity, operator, regularization, step_size,
                               regularization_weight, device, dtype)

        match = match.to(device)
        match = match.type(dtype)

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
            energy += self.reg_weight * self.regularization(self.field - self.identity).sum()

        return energy

    def update(self, update_field):
        # Apply the step size to the update field
        update_field = update_field * self.step_size

        # The field stored in self is always a look up table, so
        self.field = (self.field - self.identity) + update_field

        # Change the field back to an lut and apply to the moving image
        self.field = self.field + self.identity

        self.moving = ApplyGrid(self.field)(self.source)

    def step(self):

        # Calculate the similarity body force
        body_v = self.similarity.c1(self.target, self.moving, self.field)

        # Apply the operator to the body force
        body_v = self.operator(body_v)

        if self.regularization:
            reg_v = self.reg_weight * self.regularization.c1(self.field - self.identity)
            body_v = body_v - reg_v

        # Update variables
        self.update(body_v)

        # Calculate and return the new energy
        new_energy = self.energy()
        return new_energy

    def get_field(self):
        return self.field

    def get_image(self):
        return self.moving
