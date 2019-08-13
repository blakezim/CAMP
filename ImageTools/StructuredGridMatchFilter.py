import torch
from CAMP.ImageOperators import ApplyGrid, Gradient
from CAMP.Core.StructuredGridClass import StructuredGrid

from ._BaseTool import Filter


# TODO Multiscale matching
# TODO make the smoothing and regularizer use the same kernel
# TODO Use only one applier


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
        self.identity = StructuredGrid.FromGrid(source)
        self.identity.set_to_identity_lut_()
        self.initial_energy = self.energy()

        self.gradients = Gradient.Create(dim=len(source.size), device=device, dtype=dtype)(source)
        self.moving_grads = ApplyGrid(self.field)(self.gradients)

    @staticmethod
    def Create(source, target, similarity, operator, regularization=None, step_size=0.001,
               regularization_weight=0.1, incompressible=True, device='cpu', dtype=torch.float32):

        match = IterativeMatch(source, target, similarity, operator, regularization, step_size,
                               regularization_weight, incompressible, device, dtype)

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
            reg_e = 0.25 * self.reg_weight * self.regularization(self.field - self.identity, self.operator).sum()

        return [energy.item(), reg_e.item(), (reg_e + energy).item()]
        # return energy + reg_e

    def step(self):

        self.moving = ApplyGrid(self.field)(self.source)
        self.moving_grads = ApplyGrid(self.field)(self.gradients)

        # Calculate the similarity body force
        body_v = self.similarity.c1(self.target, self.moving, self.moving_grads)

        # Apply the operator to the body force
        body_v = self.operator.apply_inverse(body_v)

        if self.regularization:
            reg_v = self.reg_weight * self.regularization.c1(self.field - self.identity, self.operator)
            body_v = body_v + reg_v

        if self.incompressible:
            body_v = self.operator.project_incompressible(body_v)

        # Update the field
        self.field = ((self.field - self.identity) - self.step_size*body_v) + self.identity

        # Update variables
        self.moving = ApplyGrid(self.field)(self.source)

        # Calculate and return the new energy
        new_energy = self.energy()
        return new_energy

    def multi_scale(self, scale=[2, 1], niter=[150, 25], step=[0.4, 0.01], regw=[0.2, 0.05]):

        # Need to make a copy of the source image
        original_source = self.source.clone()
        original_target = self.target.clone()
        original_grads = self.gradients.clone()
        energy = [[] for _ in range(0, len(scale))]

        for i, s in enumerate(scale):
            print(f'Scale: {s}, N Iter: {niter[i]}, Step: {step[i]}, Reg Weight:{regw[i]}')

            # self.moving = self.source.clone()

            # Reset the image
            if i > 0:
                self.source = original_source.clone()
                self.target = original_target.clone()
                self.gradients = original_grads.clone()

            # Set the size of the field
            if i < len(scale) - 1:
                self.source.set_size(original_source.size // s, inplace=True)
                # self.moving.set_size(original_source.size // s, inplace=True)
                self.target.set_size(original_source.size // s, inplace=True)
                self.gradients.set_size(original_source.size // s, inplace=True)

            self.moving.set_size(original_source.size // s, inplace=True)
            self.field.set_size(original_source.size // s, inplace=True)
            self.identity.set_size(original_source.size // s, inplace=True)
            # self.identity.set_to_identity_lut_()

            # Need to update the size of the operator
            self.operator = self.operator.set_size(self.moving)
            self.moving = ApplyGrid(self.field)(self.source)

            self.reg_weight = regw[i]
            self.step_size = step[i]

            energy[i] = [self.energy()]
            print(f'Iteration: 0   Energy: {self.energy()}')

            for it in range(1, niter[i]+1):
                energy[i].append(self.step())

                if it % 10 == 0:
                    print(f'Iteration: {it}   Energy: {energy[i][-1]}')

        self.target = original_target.clone()
        self.source = original_source.clone()
        self.moving = ApplyGrid(self.field)(self.source)

        return energy

    def get_field(self):
        return self.field

    def get_image(self):
        return self.moving
