import torch
import SimpleITK as sitk

from CAMP.Core import *


def LoadITKFile(filename, device='cpu', dtype=torch.float32):

    itk_image = sitk.ReadImage(filename)

    # ITK ordering is x, y, z. But numpy is z, y, x
    image_size = torch.as_tensor(itk_image.GetSize()[::-1], dtype=dtype)
    image_spacing = torch.as_tensor(itk_image.GetSpacing()[::-1], dtype=dtype)
    image_origin = torch.as_tensor(itk_image.GetOrigin()[::-1], dtype=dtype)
    channels = itk_image.GetNumberOfComponentsPerPixel()
    tensor = torch.as_tensor(sitk.GetArrayFromImage(itk_image))

    image_origin = image_origin[image_size != 1.0]
    image_spacing = image_spacing[image_size != 1.0]
    image_size = image_size[image_size != 1.0]
    tensor = tensor.squeeze()

    # Make sure that the channels are accounted for
    if channels == 1:
        tensor = tensor.view(channels, *tensor.shape)
    else:
        tensor = tensor.permute([-1] + torch.arange(0, len(image_size)).tolist())

    out = StructuredGrid(
        size=image_size,
        spacing=image_spacing,
        origin=image_origin,
        device=device,
        dtype=dtype,
        tensor=tensor,
        channels=channels
    )

    return out


def SaveITKFile(grid, f_name):

    dim = len(grid.size)
    size = [1] * 3
    size[0:dim] = [int(x) for x in grid.size]
    shape = [grid.channels] + size

    # Need to put the vector in the last dimension
    vector_grid = grid.data.view(shape).permute(1, 2, 3, 0)  # it will always be this size now

    itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy())

    spacing = [1.0] * 3
    spacing[0:dim] = grid.spacing.tolist()
    origin = [0.0] * 3
    origin[0:dim] = grid.origin.tolist()

    # ITK ordering is x, y, z. But numpy is z, y, x
    itk_image.SetSpacing(spacing[::-1])
    itk_image.SetOrigin(origin[::-1])

    sitk.WriteImage(itk_image, f_name)

