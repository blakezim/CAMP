import torch

from ..UnaryOperators.ApplyGridFilter import ApplyGrid
from ...Core.StructuredGridClass import StructuredGrid
from ._BinaryFilter import Filter


class ComposeGrids(Filter):
    def __init__(self, padding_mode='border', device='cpu', dtype=torch.float32):
        super(ComposeGrids, self).__init__()

        self.device = device
        self.dtype = dtype
        self.padding_mode = padding_mode

    @staticmethod
    def Create(padding_mode='border', device='cpu', dtype=torch.float32):
        """
        Object to compose :class:`StructuredGrid` look-up table fields into one grid.

        :param pad_mode: padding mode for outside grid values - one of 'zeros', 'border', or 'reflection'.
            Default: 'zeros'
        :type pad_mode: str
        :param device: Memory location - one of 'cpu', 'cuda', or 'cuda:X' where X specifies the device identifier.
            Default: 'cpu'
        :type device: str
        :param dtype: Data type for the attributes. Specified from torch memory types. Default: 'torch.float32'
        :type dtype: str
        :return: Object to compose a list of look-up tables.
        """
        comp = ComposeGrids(padding_mode, device, dtype)
        comp = comp.to(device)
        comp = comp.type(dtype)

        return comp

    def forward(self, L):

        # TODO need to fix the enumerate method so that 'f' is on GPU
        
        """
        Given a list of :class:`StructuredGrid` look-up tables L = [L0, L1, L2] returns a composed look-up table
        comp_field = L2(L1(L0(x))) of type :class:`StructuredGrid`.
        
        :param L: List of look-up tables. All fields in the list must be on the same memory device.
        :type L: list, tuple
        :return: Composed look-up tables Comp_filed
        """

        comp_field = StructuredGrid.FromGrid(L[0])
        comp_field.set_to_identity_lut_()

        for f in L:
            comp_field = ApplyGrid.Create(
                comp_field,
                device=self.device,
                dtype=self.dtype,
                pad_mode=self.padding_mode
            )(f, comp_field).clone()

        return comp_field
