import numpy as np
# import Classes as cc
from . import *
# import scipy.interpolate as interp


class DisplayException(Exception):
    """exception for this class"""
    pass


# Function to determine if the image is 3D or not
def _is_3d(im, color):
    if color:
        if len(im.size()[1:]) == 3:
            return True
        else:
            return False
    else:
        if len(im.size()) == 3:
            return True
        else:
            return False


def GetSliceIndex(Image, dim):

    if dim == 'z':
        if Image.isColor():
            return Image.data.size()[1] // 2
        else:
            return Image.data.size()[0] // 2

    if dim == 'x':
        if Image.isColor():
            return Image.data.size()[2] // 2
        else:
            return Image.data.size()[1] // 2

    if dim == 'y':
        if Image.isColor():
            return Image.data.size()[3] // 2
        else:
            return Image.data.size()[2] // 2


def ExtractImageSlice(Image, dim, sliceIdx, color):

    def _get_slice_index(im, dim, color):

        if dim == 'z':
            if color:
                return im.size[1] // 2
            else:
                return Image.data.size()[0] // 2

        if dim == 'x':
            if color:
                return Image.data.size()[2] // 2
            else:
                return Image.data.size()[1] // 2

        if dim == 'y':
            if color:
                return Image.data.size()[3] // 2
            else:
                return Image.data.size()[2] // 2

    def _get_slice(im, dim, sliceIdx, color):
        if not sliceIdx:
            sliceIdx = _get_slice_index(Image, dim, color)

        if dim == 'z':
            if color:
                return im[:, sliceIdx, :, :].squeeze()
            else:
                return im[sliceIdx, :, :].squeeze()

        if dim == 'x':
            if color:
                return im[:, :, sliceIdx, :].squeeze()
            else:
                return im[:, sliceIdx, :].squeeze()

        if dim == 'y':
            if color:
                return im[:, :, :, sliceIdx].squeeze()
            else:
                return im[:, :, sliceIdx].squeeze()

    # Check what type was passed
    if type(Image).__name__ == 'Tensor':
        im = Image.copy()  # Make sure we don't mess with the original tensor
        return _get_slice(im, dim, sliceIdx, color)

    elif type(Image).__name__ == 'Image':
        im = Image.data.clone()  # Make sure we don't mess with the original tensor
        # Need a function to return a new Image with proper origin and spacing
        return Image(im)


def GetAspect(Image, axis='default', retFloat=True):
    '''given a image grid, determines the 2D aspect ratio of the
    off-dimension'''

    imsz = Image.size.tolist()

    # if color:
    #     # aspect is always (displayed) height spacing/width spacing
    #     # sz is displayed (displayed, apparent) [width, height]
    #     if dim == 'x':
    #         aspect = Image.spacing[2]/Image.spacing[0]
    #         sz = [imsz[0], imsz[2]]
    #     elif dim == 'y':
    #         aspect = Image.spacing[1]/Image.spacing[0]
    #         sz = [imsz[0], imsz[1]]
    #     else:
    #         aspect = Image.spacing[1]/Image.spacing[2]
    #         sz = [imsz[2], imsz[1]]
    # else:

    # Might need to flip these two
    aspect = (Image.spacing[0] / Image.spacing[1]).item()
    sz = [imsz[1], imsz[0]]

    if axis == 'cart':
        aspect = 1.0/aspect
        sz = sz[::-1]

    if retFloat:
        return aspect

    if aspect > 1:
        scale = [sz[0]/aspect, sz[1]*1.0]
    else:
        scale = [sz[0]*1.0, sz[1]*aspect]

    # scale incorporates image size (grow if necessary) and aspect ratio
    while scale[0] <= 400 and scale[1] <= 400:
        scale = [scale[0]*2, scale[1]*2]

    return [int(round(scale[0])), int(round(scale[1]))]


def DispImage(Image, rng=None, cmap='gray', title=None,
              new_figure=True, color=False, colorbar=True, axis='default',
              dim=0, slice_index=None):
    """
    :param Image: Input Image ([RGB[A]], [Z], Y, X)
    :param rng: Intensity range (list, tuple) (defaults to data intensity limits)
    :param cmap: Matplotlib colormap ('gray', 'jet', etc)
    :param title: Figure Title (string)
    :param newFig: Create a new figure (bool)
    :param colorbar: Display colorbar (bool)
    :param axis: Axis direction.  'default' has (0,0) in the upper left hand corner
                                      and the x direction is vertical
                                 'cart' has (0,0) in the lower left hand corner
                                      and the x direction is horizontal
    :param dim: Dimension along which to plot ([0], 1, 2)
    :param sliceIdx: Slice index along 'dim' to plot
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.ion()  # tell it to use interactive mode -- see results immediately

    if type(Image).__name__ != 'StructuredGrid':
        raise RuntimeError(
            f'Can only plot StructuredGrid types - received {type(Image).__name__}'
        )

    # Make sure the image is only 2D at this point
    if len(Image.size) == 3:
        if not slice_index:
            slice_index = int(Image.size[0].item() // 2)
        Image = Image.extract_slice(slice_index, dim)

    # Get the aspect ratio of the image
    aspect = GetAspect(Image, axis=axis, retFloat=True)

    # Define a function to display a tensor
    # This allows us to pass both an Image and a Tensor type into DispImage
    # def _display_tensor(tensor, rng=None, cmap='gray', title=None, new_figure=True, colorbar=True,
    #                     axis='default', dim='z', color=False, aspect=[1, 1]):

    im = Image.data.to('cpu').detach().clone()  # Make sure that the tensor is on the CPU and detached

    # create the figure if requested
    if new_figure:
        fig = plt.figure()
    plt.clf()  # don't be slow, also clear colorbars if necessary
    if rng is None:
        mm = [im.min().item(), im.max().item()]
        vmin = mm[0]
        vmax = mm[1]
        if mm[1] - mm[0] == 0:
            vmin -= 1
            vmax += 1
    else:
        vmin, vmax = rng

    # if color:  # for color images
    #     # color images need to be in 0-1 range
    #     if im.max(0) > 1.0:
    #         denom = max(2.0 ** (np.ceil(np.log2(im.max(0)))), 1.0)  # nextpow2
    #     else:
    #         denom = 1.0
    #     imnp = im.numpy() / denom
    # else:
    imnp = im.squeeze().numpy()

    # If it is 3D now, then it is color

    if np.isnan(imnp).any():
        raise DisplayException("DispImage: Image contains NaNs, cannot plot")

    if axis == 'default':
        img = plt.imshow(imnp, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, origin='lower')

    else:  # axis == 'cart'
        if color:  # for color images
            img = plt.imshow(np.squeeze(imnp.transpose(2, 1, 0)),
                             cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
        else:
            img = plt.imshow(np.squeeze(imnp.transpose()),
                             cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
        plt.gca().invert_yaxis()

    plt.xticks([])  # no ticks
    plt.yticks([])
    plt.axis('off')  # no border
    if title is not None:
        plt.title(title)
    img.set_interpolation('nearest')

    if colorbar:
        plt.colorbar()

    plt.draw()
    plt.autoscale()


# def DisplayJacobianDeterminant(Field, rng=None, cmap='jet', title=None, newFig=True, colorbar=True):
#
#     if type(Field).__name__ != 'StructuredGrid':
#         raise RuntimeError(
#             f'Can only plot StructuredGrid types - received {type(Field).__name__}'
#         )


def DispFieldGrid(Field, grid_size=None, title=None,
                  newFig=True, fig_len=7, dim='z', axis=None,
                  slice_index=None):
    """Displays a grid of the displacement field h
    Assumed to be ([dim], [Z], Y, X)"""

    import matplotlib.pyplot as plt
    plt.ion()  # tell it to use interactive mode -- see results immediately

    if type(Field).__name__ != 'StructuredGrid':
        raise RuntimeError(
            f'Can only plot StructuredGrid types - received {type(Field).__name__}'
        )

    # Make sure the image is only 2D at this point
    if len(Field.size) == 3:
        if not slice_index:
            slice_index = Field.size[0] // 2
        Field = Field.extract_slice(slice_index, dim)

    # It will be 2D at this point
    # spx = Field.spacing[-1].item()
    # spy = Field.spacing[-2].item()

    field = Field.data.clone()
    # Change the field to be between -1 and 1
    field = field - Field.origin.view(*Field.size.shape, *([1] * len(Field.size)))
    field = field / (Field.spacing * (Field.size / 2)).view(*Field.size.shape, *([1] * len(Field.size)))
    field = field - 1

    field_y = field[-1].cpu().detach().squeeze().numpy()  # X Coordinates
    field_x = field[-2].cpu().detach().squeeze().numpy()  # Y Coordinates

    sy = Field.size[-1].item()
    sx = Field.size[-2].item()

    # hx *= spx # The fields should already be in real space
    # hy *= spy

    # realsx = (sx - 1) * spx
    # realsy = (sy - 1) * spy
    #
    if newFig:
        grid = plt.figure(figsize=(fig_len, fig_len), dpi=140)
        grid.set_facecolor('white')
    else:
        plt.clf()
    if title is not None:
        plt.title(title)

    # dont allow for more than 127 lines
    if grid_size is None:
        grid_sizex = max(sx//64, 1)
        grid_sizey = max(sy//64, 1)
    else:
        grid_sizex = grid_size
        grid_sizey = grid_size

    grid_sizex = int(grid_sizex)
    grid_sizey = int(grid_sizey)

    # This may not always be right
    hx_sample_h = field_x[grid_sizex//2::grid_sizex, :]
    hy_sample_h = field_y[grid_sizex//2::grid_sizex, :]

    hx_sample_v = field_x[:, grid_sizey//2::grid_sizey]
    hy_sample_v = field_y[:, grid_sizey//2::grid_sizey]

    # keep the figure square, but make sure the whole grid fits in the fig
    # These fields should always range between [-1, 1]
    minax = -1
    maxax = 1

    plt.axis([minax, maxax, maxax, minax])

    # plot horizontal lines (y values)
    plt.plot(hy_sample_h.transpose(), hx_sample_h.transpose(), 'k')
    # plot vertical lines (x values)
    plt.plot(hy_sample_v, hx_sample_v, 'k')
    # make grid look nicer
    plt.axis('off')

    plt.draw()

    ## DISPLAY WITH ASPECT
    # field = interday_slice.data.clone()
    # # field = field - interday_slice.origin.view(*interday_slice.size.shape, *([1] * len(interday_slice.size)))
    # # field = field / (interday_slice.spacing * (interday_slice.size / 2)).view(*interday_slice.size.shape,
    # #                                                                   *([1] * len(interday_slice.size)))
    # # field = field - 1
    # field_y = field[-1].cpu().detach().squeeze().numpy()  # X Coordinates
    # field_x = field[-2].cpu().detach().squeeze().numpy()  # Y Coordinates
    # sy = interday_slice.size[-1].item()
    # sx = interday_slice.size[-2].item()
    # grid_sizex = max(sx // 64, 1)
    # grid_sizey = max(sy // 64, 1)
    # grid_sizex = int(grid_sizex)
    # grid_sizey = int(grid_sizey)
    # hx_sample_h = field_x[grid_sizex // 2::grid_sizex, :]
    # hy_sample_h = field_y[grid_sizex // 2::grid_sizex, :]
    # hx_sample_v = field_x[:, grid_sizey // 2::grid_sizey]
    # hy_sample_v = field_y[:, grid_sizey // 2::grid_sizey]
    # # minax = -1.0
    # # maxax = 1.0
    # last_point = ((interday_slice.size * interday_slice.spacing) + interday_slice.origin).tolist()
    # rng_x = [interday_slice.origin.tolist()[0], last_point[0]]
    # rng_y = [interday_slice.origin.tolist()[1], last_point[1]]
    #
    # # STOP Make the field arrays
    # fig = plt.figure()
    # plt.axis([rng_y[0], rng_y[1], rng_x[1], rng_x[0]])
    # plt.plot(hy_sample_h.transpose(), hx_sample_h.transpose(), 'k')
    # plt.plot(hy_sample_v, hx_sample_v, 'k')
    # plt.axes().set_aspect((1 / (abs(rng_x[1] - rng_x[0]) / abs(rng_y[0] - rng_y[1]))) * (sx / sy))
    # plt.axis('off')
    ## END

#     def _get_info(h, slice_index, dim, Field=None):
#
#         h = h.to('cpu').detach()  # Make sure that the tensor is on the CPU and detached
#
#         if len(h.size()) == 4:
#
#             if dim == 'z':
#                 if not slice_index:
#                     slice_index = h.size()[0] // 2
#                 h = h[slice_index, :, :].squeeze()
#                 hx = h[:, :, 1].numpy()
#                 hy = h[:, :, 2].numpy()
#                 sx = h.size()[0]
#                 sy = h.size()[1]
#                 if Field:
#                     spx = Field.spacing()[1].item()
#                     spy = Field.spacing()[2].item()
#             elif dim == 'y':        # not tested
#                 if not slice_index:
#                     slice_index = h.size()[2] // 2
#                 h = h[:, :, slice_index].squeeze()
#                 hx = h[:, :, 0].numpy()
#                 hy = h[:, :, 1].numpy()
#                 sx = h.size()[0]
#                 sy = h.size()[1]
#                 if Field:
#                     spx = Field.spacing()[0].item()
#                     spy = Field.spacing()[1].item()
#             elif dim == 'x':        # not tested
#                 if not slice_index:
#                     slice_index = h.size()[1] // 2
#                 h = h[:, slice_index, :].squeeze()
#                 hx = h[:, :, 0].numpy()
#                 hy = h[:, :, 2].numpy()
#                 sx = h.size()[0]
#                 sy = h.size()[1]
#                 if Field:
#                     spx = Field.spacing()[0].item()
#                     spy = Field.spacing()[2].item()
#
#         else:
#             hx = h.numpy()[:, :, 0]
#             hy = h.numpy()[:, :, 1]
#             sx = h.size()[0]
#             sy = h.size()[1]
#             if Field:
#                 spx = Field.spacing()[0].item()
#                 spy = Field.spacing()[1].item()
#
#         if 'spx' not in locals():
#             spx = 1.0
#             spy = 1.0
#
#         return hx, hy, sx, sy, spx, spy
#
#     if type(Field).__name__ != 'Stru':
#         h = Field.h.clone()  # Make sure we don't mess with the original tensor
#         hy, hx, sx, sy, spx, spy = _get_info(h, slice_index, dim, Field) # x,y flipped for plotting
#
#     # account for image spacing
#     hx *= spx
#     hy *= spy
#
#     realsx = (sx - 1) * spx
#     realsy = (sy - 1) * spy
#
#     if axis == 'cart':
#         hx, hy = hy.T, hx.T
#     elif axis is not None:
#         raise DisplayException("Unknown value for 'axis'")
#
#     # create the figure if requested
#     # TODO: Non square grid plots
#     if newFig:
#         grid = plt.figure(figsize=(fig_len, fig_len), dpi=140)
#         grid.set_facecolor('white')
#     else:
#         plt.clf()
#     if title is not None:
#         plt.title(title)
#
#     # dont allow for more than 127 lines
#     if grid_size is None:
#         grid_sizex = max(sx//64, 1)
#         grid_sizey = max(sy//64, 1)
#     else:
#         grid_sizex = grid_size
#         grid_sizey = grid_size
#
#     # This may not always be right
#     hx_sample_h = hx[grid_sizex//2::grid_sizex, :]
#     hy_sample_h = hy[grid_sizex//2::grid_sizex, :]
#
#     hx_sample_v = hx[:, grid_sizey//2::grid_sizey]
#     hy_sample_v = hy[:, grid_sizey//2::grid_sizey]
#
#
#     # keep the figure square, but make sure the whole grid fits in the fig
#     # These fields should always range between [-1, 1]
#     minax = -1
#     maxax = 1
#
#     if axis == 'cart':
#         plt.axis([minax, maxax, minax, maxax])
#     else:  # flip y axis
#         plt.axis([minax, maxax, maxax, minax])
#
#     # plot horizontal lines (y values)
#     plt.plot(hy_sample_h.transpose(), hx_sample_h.transpose(), 'k')
#     # plot vertical lines (x values)
#     plt.plot(hy_sample_v, hx_sample_v, 'k')
#     # make grid look nicer
#     plt.axis('off')
#
#     plt.draw()
# #     plt.show()


def EnergyPlot(energy, title='Energy', new_figure=True, legend=None):
    """Plots energies

    The energies should be in the form
    [E1list, E2list, E3list, ...]
    with the legend as
    [E1legend, E2legend, E3legend]
    """
    import matplotlib.pyplot as plt
    # plt.ion()  # tell it to use interactive mode -- see results immediately
    # energy should be a list of lists, or just a single list
    if new_figure:
        plt.figure()
    plt.clf()
    en = np.array(energy)
    plt.plot(en.T)
    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)
    plt.draw()


def PlotSurface(verts, faces, fig=None, norms=None, cents=None, ax=None, color=[0, 0, 1]):

    def scale_normals(norms):
        return (norms / np.sqrt((norms ** 2).sum(1))[:, None]) / 10

    def calc_centers(tris):
        return (1 / 3.0) * np.sum(tris, 1)

    def get_colors(faces, color):
        return color[faces].mean(2) / 255.0

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not fig:
        fig = plt.figure()

    verts = verts.detach().cpu().clone().numpy()
    faces = faces.detach().cpu().clone().numpy()

    if not ax:
        ax = fig.add_subplot(111, projection='3d')

        # Determine the min and max for the axis limits
        lims = np.vstack((verts.min(0), verts.min(0), verts.max(0), verts.max(0)))
        ax.set_xlim(lims.min(0)[0] - 1, lims.max(0)[0] + 1)
        ax.set_ylim(lims.min(0)[1] - 1, lims.max(0)[1] + 1)
        ax.set_zlim(lims.min(0)[2] - 1, lims.max(0)[2] + 1)

    mesh = Poly3DCollection(verts[faces])  # Create the mesh to plot
    mesh.set_alpha(0.4)  # Set the transparency of the surface

    # Plot the normals
    if norms is not None and cents is None:
        norms = norms.detach().cpu().clone().numpy()
        norms = scale_normals(norms)
        cents = calc_centers(verts[faces])
        ax.quiver3D(cents[:, 0], cents[:, 1], cents[:, 2], norms[:, 0], norms[:, 1], norms[:, 2])

    elif norms is not None and cents is not None:
        norms = norms.detach().cpu().clone().numpy()
        norms = scale_normals(norms)
        cents = cents.detach().cpu().clone().numpy()
        ax.quiver3D(cents[:, 0], cents[:, 1], cents[:, 2], norms[:, 0], norms[:, 1], norms[:, 2])

    if len(color) != 3:
        color = color.detach().cpu().clone().numpy()
        color = get_colors(faces, color)

    mesh.set_facecolor(color)  # Set the color of the surface
    ax.add_collection3d(mesh)  # Add the mesh to the axis

    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)

    return mesh, fig, ax

