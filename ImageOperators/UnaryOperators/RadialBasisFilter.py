import torch
import numbers
import torch.nn.functional as F

from CAMP.Core.StructuredGridClass import StructuredGrid
from CAMP.ImageOperators.BinaryOperators import ComposeGrids
from ._UnaryFilter import Filter
from .ApplyGridFilter import ApplyGrid
from .FluidKernelFilter import FluidKernel


# TODO Need a way to store the grid so it can be applied to other volumes


class RadialBasis(Filter):
    def __init__(self, target_landmarks, source_landmarks, sigma=0.001, device='cpu', dtype=torch.float32):
        super(RadialBasis, self).__init__()

        if target_landmarks.shape != source_landmarks.shape:
            raise RuntimeError(
                'Shape of target and source landmarks do not match: '
                f' Target Shape: {target_landmarks.shape}, Source Shape: {source_landmarks.shape}'
            )

        self.device = device
        self.dtype = dtype
        self.num_landmarks = len(source_landmarks)
        self.dim = len(source_landmarks[0])
        self.rbf = None

        if isinstance(sigma, numbers.Number):
            sigma = torch.as_tensor([sigma] * self.dim)
        else:
            sigma = torch.as_tensor(sigma)

        self.register_buffer('sigma', sigma)
        # self.sigma = self.sigma.to(device)
        # self.sigma = self.sigma.type(dtype)

        # self.source_landmarks = source_landmarks
        # self.target_landmarks = target_landmarks

        self.register_buffer('source_landmarks', source_landmarks)
        self.register_buffer('target_landmarks', target_landmarks)
        # self.incomp = incompressible
        #
        # if incompressible:
        #     self.operator = FluidKernel.Create(
        #         accu,
        #         device=self.device,
        #         alpha=1.0,
        #         beta=0.0,
        #         gamma=0.001,
        #     )

    @staticmethod
    def Create(target_landmarks, source_landmarks, sigma=0.01, device='cpu', dtype=torch.float32):
        rbf = RadialBasis(target_landmarks, source_landmarks, sigma, device, dtype)
        rbf = rbf.to(device)
        rbf = rbf.type(dtype)

        # Can't add StructuredGrid to the register buffer, so we need to make sure they are on the right device
        for attr, val in rbf.__dict__.items():
            if type(val).__name__ == 'StructuredGrid':
                val.to_(device)
                val.to_type_(dtype)
            else:
                pass

        rbf._solve_affine()
        rbf._solve_matrix_spline()

        return rbf

    def _solve_matrix_spline(self):
        def _sigma(x1, x2):
            mat = torch.eye(len(x1), device=self.device)

            diff = (x2.float() - x1.float())
            # eps = torch.tensor(1e-9, device=self.device, dtype=self.dtype)
            # r = diff * torch.log(torch.max(torch.sqrt(diff), eps))

            # r = torch.sqrt(1 + (self.sigma * (diff ** 2).sum()))
            if self.rbf == 'gaussian':
                r = (-1 * (self.sigma ** 2) * ((x1 - x2) ** 2).sum()).exp()

            else:
                r = torch.sqrt(1 + (self.sigma * (diff ** 2).sum()))
            mat = mat * r
            return mat

        dim = self.dim
        b = torch.zeros(((self.num_landmarks * self.dim), (self.num_landmarks * self.dim)), device=self.device)

        for i in range(self.num_landmarks):
            for j in range(i, self.num_landmarks):
                b[(i * dim):(i * dim) + dim, (j * dim):(j * dim) + dim] = _sigma(self.target_landmarks[i],
                                                                                 self.target_landmarks[j])
                if i != j:
                    b[(j * dim):(j * dim) + dim, (i * dim):(i * dim) + dim] = b[(i * dim):(i * dim) + dim,
                                                                              (j * dim):(j * dim) + dim]

        c = (self.tform_landmarks - self.target_landmarks).view(-1)
        # c = self.affine_landmarks.view(-1)
        x = torch.matmul(b.inverse(), c)
        self.params = x.view(self.num_landmarks, dim)

    def _solve_affine(self, rigid=False):

        source_landmarks_centered = self.source_landmarks - self.source_landmarks.mean(0)
        target_landmarks_centered = self.target_landmarks - self.target_landmarks.mean(0)

        # Solve for the transform between the points
        self.tform = torch.matmul(
            torch.matmul(
                target_landmarks_centered.t(), source_landmarks_centered
            ),
            torch.matmul(
                source_landmarks_centered.t(), source_landmarks_centered
            ).inverse()
        )

        if rigid:
            u, _, vt = torch.svd(self.tform)
            self.tform = torch.matmul(u, vt.t())

        # Solve for the translation
        self.translation = self.target_landmarks.mean(0) - torch.matmul(self.tform,
                                                                        self.source_landmarks.mean(0).t()).t()

        self.tform_landmarks = torch.matmul(self.tform, self.source_landmarks.t()).t().contiguous()
        self.tform_landmarks = self.tform_landmarks + self.translation

        # self.rigid_landmarks = torch.matmul(self.rigid, self.source_landmarks.t()).t().contiguous()
        # self.rigid_landmarks = self.rigid_landmarks + self.translation

    def _apply_affine(self, x):

        affine = torch.eye(4, device=self.device, dtype=self.dtype)

        affine[0:self.dim, 0:self.dim] = self.tform
        affine[0:self.dim, self.dim] = self.translation
        affine = affine.inverse()
        a = affine[0:self.dim, 0:self.dim]
        t = affine[-0:self.dim, self.dim]

        x.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                              x.data.permute(list(range(1, self.dim + 1)) + [0]).unsqueeze(-1))
        x.data = (x.data.squeeze() + t).permute([-1] + list(range(0, self.dim)))

        return x

    def _sample_coords(self, field):
        # We are given a vector field
        out_points = []
        for point in self.target_landmarks:
            # Point is in real space

            # Given point is in z, y, x
            index_point = (point - field.origin)/field.spacing
            torch_point = torch.as_tensor((index_point / (field.size / 2) - 1), device=self.device, dtype=self.dtype)

            # We turn it into a torch coordinat, but now grid sample is expecting x, y, z
            torch_point = torch.as_tensor((torch_point.tolist()[::-1]), device=self.device, dtype=self.dtype)
            out_point = F.grid_sample(field.data.unsqueeze(0), torch_point.view([1] * (self.dim + 1) + [self.dim]))
            out_points.append(out_point.squeeze())

            # Returned point should be z, y, x because that is what is stored in the lut

        out_points = torch.stack(out_points, 0)
        return out_points

    def _solve_vector_field(self, x):

        temp = StructuredGrid.FromGrid(x, channels=self.dim)
        accu = temp.clone() * 0.0

        for i in range(self.num_landmarks):
            temp.set_to_identity_lut_()

            point = self.target_landmarks[i].view([self.dim] + self.dim * [1]).float()
            weight = self.params[i].view([self.dim] + self.dim * [1]).float()

            sigma = self.sigma.view([self.dim] + [1] * self.dim)

            # eps = torch.tensor(1e-9, device=self.device, dtype=self.dtype)
            # diff = ((temp - point) ** 2).data
            # temp.data = diff * torch.log(torch.max(diff.sqrt(), eps))

            if self.rbf == 'gaussian':
                temp.data = -1 * (sigma ** 2) * ((temp - point) ** 2).data.sum(0, keepdim=True)
                temp.data = torch.exp(temp.data)

            else:
                temp.data = torch.sqrt(1 + sigma * ((temp - point) ** 2).data.sum(0))

            accu.data = accu.data + temp.data * weight

        return accu

    def _apply_vector_field(self, in_grid, vec_field):
        rbf_grid = StructuredGrid.FromGrid(vec_field, channels=self.dim)
        rbf_grid.set_to_identity_lut_()
        rbf_grid.data = rbf_grid.data + vec_field.data
        rbf_grid = self._apply_affine(rbf_grid)
        rbf_image = ApplyGrid.Create(rbf_grid, device=vec_field.device, dtype=vec_field.dtype)(in_grid, vec_field)

        return rbf_image, rbf_grid

    def forward(self, in_grid, out_grid=None, apply=True):

        if out_grid is not None:
            x = out_grid.clone()
        else:
            x = in_grid.clone()

        # rbf_grid = StructuredGrid.FromGrid(x, channels=self.dim)
        # rbf_grid.set_to_identity_lut_()
        accu = self._solve_vector_field(x)

        if apply:
            return self._apply_vector_field(in_grid, accu)
        else:
            return accu
        # temp = StructuredGrid.FromGrid(x, channels=self.dim)
        # accu = temp.clone() * 0.0
        #
        # for i in range(self.num_landmarks):
        #     temp.set_to_identity_lut_()
        #
        #     point = self.target_landmarks[i].view([self.dim] + self.dim * [1]).float()
        #     weight = self.params[i].view([self.dim] + self.dim * [1]).float()
        #
        #     sigma = self.sigma.view([self.dim] + [1] * self.dim)
        #
        #     # eps = torch.tensor(1e-9, device=self.device, dtype=self.dtype)
        #     # diff = ((temp - point) ** 2).data
        #     # temp.data = diff * torch.log(torch.max(diff.sqrt(), eps))
        #
        #     if self.rbf == 'gaussian':
        #         temp.data = -1 * (sigma ** 2) * ((temp - point) ** 2).data.sum(0, keepdim=True)
        #         temp.data = torch.exp(temp.data)
        #
        #     else:
        #         temp.data = torch.sqrt(1 + sigma * ((temp - point) ** 2).data.sum(0))
        #
        #     accu.data = accu.data + temp.data * weight

        # if self.incomp:
        #
        #     operator = FluidKernel.Create(
        #         accu,
        #         device=self.device,
        #         alpha=1.0,
        #         beta=0.0,
        #         gamma=0.001,
        #     )
        #
        #     # Project the deformation into incompressible
        #     accu = operator.project_incompressible(accu)

            # This field likely no longer matches the points very well
            # Look up the target points and compare them to the source points
        # def_landmarks = self._sample_coords(accu)
        # diff = def_landmarks - self.affine_landmarks

        # if apply:
        #     rbf_grid.data = rbf_grid.data + accu.data
        #     rbf_grid = self._apply_affine(rbf_grid)
        #     x_rbf = ApplyGrid.Create(rbf_grid, device=x.device, dtype=x.dtype)(in_grid, out_grid)
        #     return x_rbf, rbf_grid
        #
        # else:
        #     return accu

    def filter_incompressible(self, in_grid, out_grid, t_step=10):

        # Store the target landmarks as we are going to change these
        target_landmarks_original = self.target_landmarks.clone()

        # Need to recompute the parameters with the rigid
        self._solve_affine(rigid=True)
        self._solve_matrix_spline()

        operator = FluidKernel.Create(
            out_grid,
            device=self.device,
            alpha=1.0,
            beta=0.0,
            gamma=0.001,
        )

        composer = ComposeGrids.Create(device=self.device, dtype=self.dtype)

        # TODO Add some sort of convergence detection
        # step=0.5

        accu = StructuredGrid.FromGrid(out_grid, channels=self.dim)
        accu.set_to_identity_lut_()
        id = accu.copy()

        for _ in range(0, t_step):

            # Do the forward
            def_vec = self.forward(in_grid, out_grid, apply=False)  # Get the vector field
            def_vec = operator.project_incompressible(def_vec)  # Project the vector field to divergent free
            # Compose the field
            accu = composer([accu, id + def_vec])

            self.target_landmarks = target_landmarks_original.clone()
            def_landmarks = self._sample_coords(accu)  # Sample the grid at the target points
            self.target_landmarks = def_landmarks.clone()  # Make the new target points the sampled points
            self._solve_matrix_spline()
            print(torch.norm(def_landmarks - self.tform_landmarks, p=2, dim=1).sum())

        # for _ in range(0, t_step - 1):
        #
        #     # NEED TO RESOLVE PARAMETERS
        #     self._solve_matrix_spline()
        #     temp_vec = self.forward(in_grid, out_grid, apply=False)
        #     temp_vec = operator.project_incompressible(temp_vec)
        #     # Need to figure out how to add it to the last one
        #     def_vec = def_vec + step*temp_vec
        #     # Return the landmarks to how they were so the vector field compare with the right points
        #     # self.target_landmarks = target_landmarks_original.clone()
        #     def_landmarks = self._sample_coords(temp_vec)
        #     self.target_landmarks = def_landmarks.clone()
        #     print(torch.norm(def_landmarks - self.tform_landmarks, p=2, dim=1).sum())

        # Return the landmarks to how they were
        self.target_landmarks = target_landmarks_original.clone()

        return self._apply_vector_field(in_grid, accu - id)
