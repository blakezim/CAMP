import numpy as np
from . import *

from StructuredGridOperators.UnaryOperators.JacobianDeterminantFilter import JacobianDeterminant


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


def _GetSliceIndex(Image, dim):

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


def _ExtractImageSlice(Image, dim, sliceIdx, color):

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


def _GetAspect(Image, axis='default', retFloat=True):

    imsz = Image.size.tolist()

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
    Display an image default with a colorbar. If the input image is 3D, it will be sliced along the dim argument. If no
    slice index is provided then it will be the center slice along dim.

    :param Image: Input Image ([RGB[A]], [Z], Y, X)
    :type Image: :class:`StructuredGrid`
    :param rng: Display intensity range. Defaults to data intensity range.
    :type rng: list, tuple
    :param cmap: Matplotlib colormap. Default 'gray'.
    :type cmap: str
    :param title: Figure Title.
    :type title: str
    :param new_figure: Create a new figure. Default True.
    :type new_figure: bool
    :param colorbar: Display colorbar. Default True.
    :type colorbar: bool
    :param axis: Axis direction. 'default' has (0,0) in the upper left hand corner and the x direction is vertical
                                 'cart' has (0,0) in the lower left hand corner and the x direction is horizontal
    :type axis: str
    :param dim: Dimension along which to plot 3D image. Default is 0 (z).
    :type dim: int
    :param slice_index: Slice index along 'dim' to plot
    :type slice_index: int
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
    aspect = _GetAspect(Image, axis=axis, retFloat=True)

    # Make sure that the tensor is on the CPU and detached
    im = Image.data.to('cpu').detach().clone()

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

    imnp = im.squeeze().numpy()

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


def DisplayJacobianDeterminant(Field, rng=None, cmap='jet', title=None, new_figure=True, colorbar=True,
                               slice_index=None, dim='z'):
    """
    Calculated and display the jacobian determinant of a field.

    :param Field: Assumed to be a :class:`StructuredGrid` LUT that defines a transformation.
    :type Field: :class:`StructuredGrid`
    :param rng: Display intensity range. Defaults to jacobian determinant intensity range.
    :type rng: list, tuple
    :param cmap: Matplotlib colormap. Default 'jet'.
    :type cmap: str
    :param title: Figure Title.
    :type title: str
    :param new_figure: Create a new figure. Default True.
    :type new_figure: bool
    :param colorbar: Display colorbar. Default True.
    :type colorbar: bool
    :param dim: Dimension along which to plot 3D image. Default is 0 (z).
    :type dim: int
    :param slice_index: Slice index along 'dim' to plot
    :type slice_index: int
    :return: None
    """

    import matplotlib.pyplot as plt
    plt.ion()  # tell it to use interactive mode -- see results immediately

    if type(Field).__name__ != 'StructuredGrid':
        raise RuntimeError(
            f'Can only plot StructuredGrid types - received {type(Field).__name__}'
        )
    Field.to_('cpu')

    if len(Field.size) == 3:
        if not slice_index:
            slice_index = Field.size[0] // 2
        Field = Field.extract_slice(slice_index, dim)

    jac_filt = JacobianDeterminant.Create()
    jacobian = jac_filt(Field)
    DispImage(jacobian, cmap=cmap, title=title, new_figure=new_figure, colorbar=colorbar, rng=rng)


def DispFieldGrid(Field, grid_size=None, title=None, newFig=True, dim='z', slice_index=None):
    """
    Displays a grid of the input field. Field is assumed to be a look-up table (LUT) of type :class:`StructuredGrid`.

    :param Field: Assumed to be a :class:`StructuredGrid` LUT that defines a transformation.
    :type Field: :class:`StructuredGrid`
    :param grid_size: Number of grid lines to plot in each direction.
    :type grid_size: int
    :param title: Figure Title.
    :type title: str
    :param newFig: Create a new figure. Default True.
    :type newFig: bool
    :param dim: Dimension along which to plot 3D image. Default is 0 ('z').
    :type dim: str
    :param slice_index: Slice index along 'dim' to plot
    :type slice_index: int
    :return: None
    """

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

    field = Field.data.clone()
    # Change the field to be between -1 and 1
    field = field - Field.origin.view(*Field.size.shape, *([1] * len(Field.size)))
    field = field / (Field.spacing * (Field.size / 2)).view(*Field.size.shape, *([1] * len(Field.size)))
    field = field - 1

    field_y = field[-1].cpu().detach().squeeze().numpy()  # X Coordinates
    field_x = field[-2].cpu().detach().squeeze().numpy()  # Y Coordinates

    sy = Field.size[-1].item()
    sx = Field.size[-2].item()

    if newFig:
        grid = plt.figure()
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


def EnergyPlot(energy, title='Energy', new_figure=True, legend=None):
    """
    Plot energies from registration functions.

    :param energy: The energies should be in the form [E1list, E2list, E3list, ...]
    :type energy: list, tuple
    :param title: Figure Title.
    :type title: str
    :param new_figure: Create a new figure. Default True.
    :type new_figure: bool
    :param legend: List of strings to be added to the legend in the form [E1legend, E2legend, E3legend, ...]
    :type legend: list
    :return: None
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


def PlotSurface(verts, faces, fig=None, norms=None, cents=None, ax=None, color=(0, 0, 1)):
    """
    Plot a triangle mesh object.

    :param verts: Vertices of the mesh object.
    :type verts: tensor
    :param faces: Indices of the mesh object.
    :type faces: tensor
    :param fig: Matplotlib figure object to plot the surface on. If one is not provided, and new one is created.
    :type fig: Maplotlib figure object
    :param norms: Normals of the mesh object.
    :type norms: tensor, optional
    :param cents: Centers of the mesh object.
    :type cents: tensor, optional
    :param ax: Matplotlib axis object to plot the surface on. If one is not provided, and new one is created.
    :type ax: Maplotlib axis object
    :param color: Plotted color of the surface. Tuple of three floats between 0 and 1 specifying RGB values.
    :type color: tuple
    :return: None
    """

    def _scale_normals(norms):
        return (norms / np.sqrt((norms ** 2).sum(1))[:, None]) / 10

    def _calc_centers(tris):
        return (1 / 3.0) * np.sum(tris, 1)

    def _get_colors(faces, color):
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
        norms = _scale_normals(norms)
        cents = _calc_centers(verts[faces])
        ax.quiver3D(cents[:, 0], cents[:, 1], cents[:, 2], norms[:, 0], norms[:, 1], norms[:, 2])

    elif norms is not None and cents is not None:
        norms = norms.detach().cpu().clone().numpy()
        norms = _scale_normals(norms)
        cents = cents.detach().cpu().clone().numpy()
        ax.quiver3D(cents[:, 0], cents[:, 1], cents[:, 2], norms[:, 0], norms[:, 1], norms[:, 2])

    if len(color) != 3:
        color = color.detach().cpu().clone().numpy()
        color = _get_colors(faces, color)

    mesh.set_facecolor(color)  # Set the color of the surface
    ax.add_collection3d(mesh)  # Add the mesh to the axis

    plt.show(block=False)
    plt.draw()
    plt.pause(0.01)

    return mesh, fig, ax

