import torch
import SimpleITK as sitk

from Core import *


def LoadITKFile(filename, device='cpu', dtype=torch.float32):
    # TODO Need to check when loading NRRD - for some reason loading Sara's volumes was backwards

    itk_image = sitk.ReadImage(filename)

    # ITK ordering is x, y, z. But numpy is z, y, x
    image_size = torch.as_tensor(itk_image.GetSize()[::-1], dtype=dtype)
    image_spacing = torch.as_tensor(itk_image.GetSpacing()[::-1], dtype=dtype)
    image_origin = torch.as_tensor(itk_image.GetOrigin()[::-1], dtype=dtype)
    channels = itk_image.GetNumberOfComponentsPerPixel()
    dataArray = sitk.GetArrayFromImage(itk_image)
    if dataArray.dtype == 'uint16':
        dataArray = dataArray.astype('int32')
    tensor = torch.as_tensor(dataArray)

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

    # Need to put the vector in the last dimension
    vector_grid = grid.data.permute(list(range(1, dim +1)) + [0]).squeeze(-1)  # it will always be this size now

    if dim == 2 and vector_grid.shape[-1] == 3:
        itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy(), isVector=True)
    # elif dim == 2:
    #     itk_image = sitk.GetImageFromArray(vector_grid.unsqueeze(-2).cpu().numpy())
    else:
        itk_image = sitk.GetImageFromArray(vector_grid.cpu().numpy())


    spacing = grid.spacing.tolist()
    if dim == 2:
        spacing = [1.0] + spacing

    origin = grid.origin.tolist()
    if dim == 2:
        origin = [1.0] + origin


    # ITK ordering is x, y, z. But numpy is z, y, x
    itk_image.SetSpacing(spacing[::-1])
    itk_image.SetOrigin(origin[::-1])

    sitk.WriteImage(itk_image, f_name)

