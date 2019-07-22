import torch
import SimpleITK as sitk

from Core import *


def LoadITKImage(filename, device='cpu', dtype=torch.float32):

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

    out = Image(
        size=image_size,
        spacing=image_spacing,
        origin=image_origin,
        device=device,
        dtype=dtype,
        tensor=tensor,
        channels=channels)

    return out
