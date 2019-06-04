import torch
from Core import *
from Operators import *

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
plt.ion()


def circle(Im, R, center=None, val=1.0):
    """Creates Image3D circle with radius and center
    using index values for x, y"""
    x, y = torch.meshgrid(torch.arange(0, Im.size(0)), torch.arange(0, Im.size(1)))
    if center is None:
        center = [(Im.size(0) - 1) / 2.0, (Im.size(1) - 1) / 2.0]
    Im = (val * ((x - center[0]) ** 2 + (y - center[1]) ** 2 < R ** 2)).to(Im.device).double()
    return Im.float()


def ellipse(Im, A, B, center=None, val=1.0):
    x, y = torch.meshgrid(torch.arange(0, Im.size(0)), torch.arange(0, Im.size(1)))
    x = x.double()
    y = y.double()
    if center is None:
        center = [(Im.size(0) - 1) / 2.0, (Im.size(1) - 1) / 2.0]
    Im = (val * ((((x - center[0]) ** 2) / (A ** 2) +
                 ((y - center[1]) ** 2) / (B ** 2)) < 1)).to(Im.device).double()

    return Im


def circle_and_elipse():

    with torch.no_grad():
        alpha = 1.0
        gamma = 0.001
        # step = 0.001

        device = 'cuda:1'

        # Gaussian blur object for the images
        # gaussian = f.GaussianSmoothing(1, 7, 1).to(device)
        test = Gaussian(1, 5, 2, device=device, dim=2)

        # Create the circle image
        circle_im = Image((256, 256), device=device)
        circle_im.t = circle(circle_im.t, 20)
        # circle_im = gaussian(circle_im)
        Display.DispImage(circle_im, title='Target')
        gauss_filt = test(circle_im)
        Display.DispImage(gauss_filt, title='Filtered')
        circle_im.set_size_((512, 512))

        # Create the ellipse image
        ellipse_im = Image((256, 256), device=device)
        ellipse_im.t = ellipse(ellipse_im.t, 15, 45)
        # ellipse_im = gaussian(ellipse_im)
        Display.DispImage(ellipse_im, title='Source')


        # test = co.GaussianSmoothing(3, 5, 0.1, dim=3)
    #     # Set the origin of one of the images so we can test resample world
    #     ellipse_im.set_origin([50, 0])
    #     test = ellipse_im.copy()
    #
    #     # f.ResampleWorld()
    #
    #     # Create the h field
    #     h = cc.Field(ellipse_im.t.size())
    #     h.to_device_('cuda:1')
    #
    #     test = h.copy()
    #
    #     match = energy.ImageMatch(ellipse_im, circle_im, h, device=device)
    #
    #     # Calculate the energy in the beginning
    #     e = []
    #     e.append(match.image_energy().item())
    #     print(f'Iteration: 0  Energy: {e[-1]}')
    #
    #     for itr in range(1, 2001):
    #
    #         # Do a step in the gradient direction
    #         e.append(match.step().item())
    #
    #         print(f'Iteration: {itr}  Energy: {e[-1]}')
    #
    # plot.EnergyPlot([e])
    #
    # field = match.get_field()
    # image = match.get_image()
    # field.h = (field.h + 1.0) * 255.0 / 2

    # temp = field.h[:,:,0].clone()
    # field.h[:,:,0] = field.h[:,:,1].clone()
    # field.h[:,:,1] = temp
    #
    # plot.DispImage(image, title='Deformed Image')
    # plot.DispHGrid(field, title='Deformation Grid')
    #
    # print('Done')


if __name__ == '__main__':
    circle_and_elipse()
