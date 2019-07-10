import torch

# from Core.ImageClass import Image
from Core.FieldClass import Field
from ImageOperators.ApplyFieldFilter import ApplyHField
# from Core.GridClass import Grid
from ._BaseMatchFilter import Filter


class IterativeMatch(Filter):
    def __init__(self, source, target, similarity, regularization=None, smoothing=None, step_size=0.001,
                 regularization_weight=0.1):
        super(IterativeMatch, self).__init__(source, target, similarity, regularization, smoothing)

        if not type(source).__name__ == 'Image':
            raise RuntimeError(
                f'Only type "Image" for source is accepted, got {type(source).__name__}'
            )

        if not type(target).__name__ == 'Image':
            raise RuntimeError(
                f'Only type "Image" for target is accepted, got {type(target).__name__}'
            )

        if not source.size == target.size:
            raise RuntimeError(
                f'Images must have the same size - Target Size: {target.size}, Source Size: {source.size}'
            )

        self.moving = source.clone()  # Clone the source so we don't mess with the original image
        self.field = Field(source.size)
        self.step_size = step_size
        self.reg_weight = regularization_weight
        self.initial_energy = self.energy()

    def energy(self):
        energy = self.similarity(self.target - self.moving).sum()

        if self.regularization:
            energy += self.reg_weight * self.regularization(self.field).sum()

        return energy

    def update(self, update_field):
        # Apply the step size to the update field
        update_field = self.step_size * update_field

        # Change the stored field to a vector field and subtract the update field
        self.field.to_v_field_()
        self.field = self.field - update_field

        # Change the feild back to an h field and apply to the moving image
        self.field.to_h_field_()
        self.moving = ApplyHField(self.field)(self.source)

    def step(self):

        # Calculate the energy?

        # Calculate the similarity body force
        body_v = self.similarity.c1(self.target, self.moving)

        if self.regularization:
            reg_v = self.reg_weight * self.regularization(body_v)
            body_v = body_v + reg_v

        # Smooth the body force if required (not sure if this is before or after....)
        if self.smoothing:
            body_v = self.smoothing(body_v)

        # Update variables
        self.update(body_v)

        # Calculate and return the new energy
        new_energy = self.energy()
        return new_energy

    def get_field(self):
        return self.field

    def get_image(self):
        return self.moving
