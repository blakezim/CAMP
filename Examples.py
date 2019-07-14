import torch
from Core import *
from ImageOperators import *
from Registration import *

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
plt.ion()


def circle(Im, R, center=None, val=1.0):
    """Creates Image3D circle with radius and center
    using index values for x, y"""
    x, y = torch.meshgrid(torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1]))
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((x - center[0]) ** 2 + (y - center[1]) ** 2 < R ** 2)).to(Im.device).float().unsqueeze(0)


def ellipse(Im, A, B, center=None, val=1.0):
    x, y = torch.meshgrid(torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1]))
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((((x - center[0]) ** 2) / (A ** 2) +
               ((y - center[1]) ** 2) / (B ** 2)) < 1)).to(Im.device).float().unsqueeze(0)


def circle_and_elipse():

    with torch.no_grad():
        alpha = 1.0
        gamma = 0.001
        step = 0.1

        device = 'cuda:1'

        # Gaussian blur object for the images
        gauss_filt = Gaussian.Create(1, 5, 1, device=device, dim=2)

        # Create the circle image
        circle_im = Image((256, 256), device=device)
        circle(circle_im, 20)
        circle_im = gauss_filt(circle_im)

        # Create the ellipse image
        ellipse_im = Image((256, 256), device=device)
        ellipse(ellipse_im, 15, 45)
        ellipse_im = gauss_filt(ellipse_im)

        # Display the images
        Display.DispImage(ellipse_im, title='Source')
        Display.DispImage(circle_im, title='Target')

        # Create the smoothing operator
        smoothing = InverseLaplacian.Create(
            circle_im,
            device=device,
            alpha=alpha,
            beta=0.0,
            gamma=gamma,
            incompresible=True
        )

        # Create the matching term
        similarity = L2(dim=2, device=device)

        # Now make the registration object
        matcher = IterativeMatch(
            source=ellipse_im,
            target=circle_im,
            similarity=similarity,
            regularization=None,
            smoothing=smoothing,
            device=device,
            step_size=step
        )

        energy = [matcher.initial_energy.item()]
        print(f'Iteration: 0   Energy: {matcher.initial_energy.item()}')
        for i in range(1, 1000):

            energy.append(matcher.step().item())

            print(f'Iteration: {i}   Energy: {energy[-1]}')

        # Get the image from the matcher
        def_image = matcher.get_image()
        Display.DispImage(def_image, title='Moving')

        print('All Done')


if __name__ == '__main__':
    circle_and_elipse()
