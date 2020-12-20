import torch

from StructuredGridOperators.UnaryOperators.ApplyGridFilter import ApplyGrid
from Core.StructuredGridClass import StructuredGrid
from ._BinaryFilter import Filter


class ComposeGrids(Filter):
    def __init__(self, padding_mode='border', device='cpu', dtype=torch.float32):
        super(ComposeGrids, self).__init__()

        self.device = device
        self.dtype = dtype
        self.padding_mode = padding_mode

    @staticmethod
    def Create(padding_mode='border', device='cpu', dtype=torch.float32):
        comp = ComposeGrids(padding_mode, device, dtype)
        comp = comp.to(device)
        comp = comp.type(dtype)

        return comp

    def forward(self, x):

        # TODO need to fix the enumerate method so that 'f' is on GPU

        temp_field = StructuredGrid.FromGrid(x[0])
        temp_field.set_to_identity_lut_()

        for f in x:
            temp_field = ApplyGrid.Create(
                temp_field,
                device=self.device,
                dtype=self.dtype,
                pad_mode=self.padding_mode
            )(f, temp_field).clone()

        return temp_field
