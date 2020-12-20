import torch
from Core import *
from StructuredGridOperators import *
from StructuredGridTools import *

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


def circle(Im, R, center=None, val=1.0):

    x, y = torch.meshgrid([torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1])])
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((x - center[0]) ** 2 + (y - center[1]) ** 2 < R ** 2)).to(Im.device).float().unsqueeze(0)


def ellipse(Im, A, B, center=None, val=1.0):
    x, y = torch.meshgrid([torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1])])
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((((x - center[0]) ** 2) / (A ** 2) +
               ((y - center[1]) ** 2) / (B ** 2)) < 1)).to(Im.device).float().unsqueeze(0)


def circle_and_elipse():

    alpha = 1.0
    beta = 0.0
    gamma = 0.001
    device = 'cuda:1'

    # Gaussian blur object for the images
    gauss_filt = Gaussian.Create(1, int(3*1.5), 1.5, device=device, dim=2)

    # # Create the circle image
    circle_im = StructuredGrid((130, 200), device=device)
    circle(circle_im, 25)
    circle_im = gauss_filt(circle_im)

    # Create the ellipse image
    ellipse_im = StructuredGrid((130, 200), device=device)
    ellipse(ellipse_im, 20, 40)
    ellipse_im = gauss_filt(ellipse_im)

    # Display the images
    DispImage(circle_im, title='Target')
    DispImage(ellipse_im, title='Source')

    # Create the smoothing operator
    smoothing = FluidKernel.Create(
        circle_im,
        device=device,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # Create the matching term
    similarity = L2Similarity.Create(dim=2, device=device)

    # Now make the registration object
    matcher_incomp = IterativeMatch.Create(
        source=ellipse_im,
        target=circle_im,
        similarity=similarity,
        regularization=None,
        operator=smoothing,
        device=device,
        step_size=0.05,
        incompressible=True,
    )

    energy_incomp = [matcher_incomp.initial_energy]
    print(f'Iteration: 0   Energy: {matcher_incomp.initial_energy}')
    for i in range(1, 10000):

        energy_incomp.append(matcher_incomp.step())
        print(f'Iteration: {i}   Energy: {energy_incomp[-1]}')

        if energy_incomp[-1] > energy_incomp[-2] - (1e-3 * energy_incomp[-2]):
            break

    # Get the image from the matcher
    def_ellipse_incomp = matcher_incomp.get_image()
    field_incomp = matcher_incomp.get_field()

    # Plot the deformed image
    DispImage(def_ellipse_incomp, title='Deformed Ellipse with Incompressibility')
    DispFieldGrid(field_incomp, title='Incompressible Grid')
    DisplayJacobianDeterminant(field_incomp, title='Jacobian Determinant Incompressibility', cmap='jet')

    # # Now do the same thing for the compressible version
    matcher_comp = IterativeMatch.Create(
        source=ellipse_im,
        target=circle_im,
        similarity=similarity,
        regularization=None,
        operator=smoothing,
        device=device,
        step_size=0.05,
        incompressible=False,
    )

    energy_comp = [matcher_comp.initial_energy]
    print(f'Iteration: 0   Energy: {matcher_comp.initial_energy}')
    for i in range(1, 100000):
        energy_comp.append(matcher_comp.step())
        if energy_comp[-1] > energy_comp[-2] - (1e-3 * energy_comp[-2]):
            break
        print(f'Iteration: {i}   Energy: {energy_comp[-1]}')

    def_ellipse_comp = matcher_comp.get_image()
    field_comp = matcher_comp.get_field()

    # Plot the deformed image
    DispImage(def_ellipse_comp, title='Deformed Ellipse')
    DispFieldGrid(field_comp, title='Incompressible Grid')
    DisplayJacobianDeterminant(field_comp, title='Jacobian Determinant', cmap='jet')

    print('All Done')


if __name__ == '__main__':
    circle_and_elipse()
