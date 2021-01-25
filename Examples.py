import torch
import torch.optim as optim
from camp.Core import *
from camp.FileIO import *
from camp.UnstructuredGridOperators.BinaryOperators.DeformableCurrentsFilter import DeformableCurrents
# from StructuredGridOperators import *
# from StructuredGridTools import *

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


def deformable_register(tar_surface, src_surface, deformable_lr=1.0e-04, currents_sigma=None, prop_sigma=None,
                        converge=0.3, device='cpu', iters=200):
    if currents_sigma is None:
        currents_sigma = [0.5]
    if prop_sigma is None:
        prop_sigma = [1.5, 1.5, 0.5]

    def _prop_gradients(prop_locations, grads, verts, prop_sigma):
        d = ((prop_locations.unsqueeze(1) - verts.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
        d = torch.exp(-d / (2 * prop_sigma[None, None, :] ** 3))
        return (grads[None, :, :].repeat(len(d), 1, 1) * d).sum(1)

    prop_sigma = torch.tensor(prop_sigma, device=device)

    for i, sigma in enumerate(currents_sigma):

        # Create the deformable model
        model = DeformableCurrents.Create(
            src_surface.copy(),
            tar_surface.copy(),
            sigma=sigma,
            kernel='cauchy',
            device=device
        )

        # Set up the optimizer
        optimizer = optim.SGD([
            {'params': [model.src_vertices], 'lr': deformable_lr[i]}], momentum=0.9, nesterov=True
        )

        # Now iterate
        energy = []
        for epoch in range(0, iters):
            optimizer.zero_grad()
            loss = model()

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.6f} ')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients

            with torch.no_grad():

                # Create a single array of the gradients to be propagated
                concat_grad = model.src_vertices.grad.clone()
                concat_vert = model.src_vertices.clone()

                # Propagate the gradients to the register surfaces
                model.src_vertices.grad = _prop_gradients(model.src_vertices, concat_grad, concat_vert, prop_sigma)

                optimizer.step()

            if epoch > 10 and np.mean(energy[-7:]) - energy[-1] < converge:
                break

        # Update the surfaces
        src_surface.vertices = model.src_vertices.detach().clone()

    return src_surface


def monkey_surface():
    device = 'cuda:1'
    # Load the two surfaces
    surface_dir = '/home/sci/blakez/code/CampExamples/'
    src_vert, src_edges = ReadOBJ(f'{surface_dir}/Monkey_Source.obj')
    tar_vert, tar_edges = ReadOBJ(f'{surface_dir}/Monkey_Target.obj')

    src_surface = TriangleMesh(src_vert, src_edges)
    tar_surface = TriangleMesh(tar_vert, tar_edges)
    src_surface.to_(device)
    src_surface.to_(device)

    params = {
        'currents_sigma': [0.5, 0.075],
        'propagation_sigma': [0.75, 0.75, 0.75],
        'deformable_lr': [2.0e-03, 1.0e-02],
        'converge': 0.00001,
        'niter': 500
    }

    def_surface = deformable_register(
        tar_surface.copy(),
        src_surface.copy(),
        currents_sigma=params['currents_sigma'],
        prop_sigma=params['propagation_sigma'],
        deformable_lr=params['deformable_lr'],
        converge=params['converge'],
        device=device,
        iters=params['niter']
    )

    WriteOBJ(def_surface.vertices, def_surface.indices, '/home/sci/blakez/test_monkey.obj')


if __name__ == '__main__':
    # circle_and_elipse()
    monkey_surface()
