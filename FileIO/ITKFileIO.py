import torch
import SimpleITK as sitk

from Core import *


def LoadITKFile(filename, device='cpu', dtype=torch.float32):

    itk_image = sitk.ReadImage(filename)

    # ITK ordering is x, y, z. But numpy is z, y, x
    image_size = itk_image.GetSize()[::-1]
    image_spacing = itk_image.GetSpacing()[::-1]
    image_origin = itk_image.GetOrigin()[::-1]
    channels = itk_image.GetNumberOfComponentsPerPixel()
    tensor = torch.as_tensor(sitk.GetArrayFromImage(itk_image))

    # Make sure that the channels are accounted for
    tensor = tensor.view(channels, *tensor.shape)

    # Need to account for if the direction is not in the orthogonal direction

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

    # Need to put the vector in the last dimension
    vector_grid = grid.data.permute(torch.arange(1, len(grid.shape())).tolist() + [0])

    itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy())

    # ITK ordering is x, y, z. But numpy is z, y, x
    itk_image.SetSpacing(grid.spacing.tolist()[::-1])
    itk_image.SetOrigin(grid.origin.tolist()[::-1])

    sitk.WriteImage(itk_image, f_name)

