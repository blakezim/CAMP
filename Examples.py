import torch
import cv2
import numpy as np
from Core import *
from ImageOperators import *
from ImageTools import *
from FileIO import *

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
plt.ion()


def circle(Im, R, center=None, val=1.0):
    """Creates Image3D circle with radius and center
    using index values for x, y"""
    x, y = torch.meshgrid(torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1]))
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((x - center[0]) ** 2 + (y - center[1]) ** 2 < R ** 2)).to(Im.device).float().unsqueeze(0)


def ellipse(Im, A, B, center=None, val=1.0):
    x, y = torch.meshgrid(torch.arange(0, Im.size[0]), torch.arange(0, Im.size[1]))
    if center is None:
        center = [(Im.size[0] - 1) / 2.0, (Im.size[1] - 1) / 2.0]
    Im.data = (val * ((((x - center[0]) ** 2) / (A ** 2) +
               ((y - center[1]) ** 2) / (B ** 2)) < 1)).to(Im.device).float().unsqueeze(0)


def solve_matrix_spline(target_landmarks, source_landmarks, sigma=0.1, incomp=False):

    def _sigma(target, source):
        r = -1 * (sigma**2) * ((source - target) ** 2).sum()
        # return r.exp()
        # r = (source.float() - target.float()).norm(2, dim=-1)

        # Tails could go to zero...
        # eps = 1e-9
        # r[r < eps] = eps

        mat = torch.eye(len(target))
        # return torch.eye(len(target)) * torch.exp(-sigma * z)
        # return torch.eye(len(target)) * torch.sqrt(1 + (sigma * z) ** 2)

        if incomp:
            # Compute the gradient
            diff = source.float() - target.float()
            # mat = mat * -1 * (4 * (sigma ** 2) * (diff ** 2) - 2 * sigma) * (-(sigma) * (diff ** 2).sum()).exp()
            # mat[0, 1] = 4 * sigma ** 2 * (-(sigma) * (diff ** 2).sum()).exp()
            # mat[1, 0] = 4 * sigma ** 2 * (-(sigma) * (diff ** 2).sum()).exp()

            grad = -2 * (sigma ** 2) * diff * (-(sigma ** 2) * (diff ** 2).sum()).exp()
            ggt = torch.ger(grad, grad)

            # gg = (-2 * (sigma ** 2) + 4 * (sigma ** 4) * (diff ** 2)) * (-(sigma ** 2) * (diff ** 2).sum()).exp()


            mat = ggt - (grad ** 2) * mat

            # mag = ((diff) ** 2).sum()
            # for i in range(dim):
            #     for j in range(dim):
            #         if i == j:
            #             mat[i, j] = 0  # torch.log(mag) + 2 * diff[i] / mag + 1
            #         else:
            #             mat[i, j] = (2 * diff[i] * diff[j]) / mag
            # mat = mat * -1
        mat = mat * r.exp()

        return mat

    if target_landmarks.shape != source_landmarks.shape:
        raise RuntimeError(
            'Shape of target and source landmarks do not match: '
            f' Target Shape: {target_landmarks.shape}, Source Shape: {target_landmarks.shape}'
        )

    shape = source_landmarks.shape

    dim = shape[-1]
    # A = torch.zeros(dim, dim)
    t = torch.zeros(dim)
    N = shape[0]

    # Create a matrix of zeros
    B = torch.zeros(((N * dim), (N * dim)))

    # test = _sigma(target_landmarks, source_landmarks)

    for i in range(N):
        for j in range(i, N):

            B[(i*dim):(i*dim) + dim, (j*dim):(j*dim)+dim] = _sigma(target_landmarks[i], target_landmarks[j])

            # B[i, j] = _sigma(source_landmarks[i], source_landmarks[j])
            if i != j:
                B[(j*dim):(j*dim)+dim, (i*dim):(i*dim) + dim] = B[(i*dim):(i*dim) + dim, (j*dim):(j*dim)+dim]

    c = torch.zeros((N * dim))
    c[0:N*dim] = source_landmarks.view(-1)

    X = torch.matmul(B.inverse(), c)

    # X, _ = torch.solve(c.unsqueeze(-1), B)

    params = X[0:N*dim].view(N, dim)
    # affine = X[N*dim:].view(dim, dim+1)
    affine = 0

    return affine, params



def SolveSpline(landmarks):
    ''' given a set of N (2D or 3D) correspondences, i.e. :
    landmarks =     [ [[x0, x1, x2], [y0, y1, y2]],
                 ...
      [[x0, x1, x2], [y0, y1, y2]] ]

    return a dict of the parameters of the radial basis function:

    'params': [ k0, k1, k2 ]    # each is an N-D vector
    'points': [ p0, p1, p2 ]    # each is an N-D vector (copied x values)
    'A': (3x3 matrix)
    't': (lenght 3 vector)

    that describe the deformation field:
    u(x) = sum_I^N k_i phi_i(x) + Ax + t

    such that phi_i(x) =
    ||x_i-x||^2 ln(||x_i-x||) (2D)
    ||x_i-x|| (3D)
    '''

    def _sigma(x1, x2, dim):
        '''radial basis function of two points'''
        z = np.linalg.norm(x1 - x2)
        if dim == 2:
            eps = 1e-9
            return z ** 2 * np.log(max(z, eps))
        else:
            return z

    # convert now to a list of numpy arrays
    landmarks = [(np.array(lm[0]), np.array(lm[1])) for lm in landmarks]

    if len(landmarks[0][0]) == 3:
        dim = 3
        # A = np.array((3, 3))
        # t = np.array(3)
    else:
        dim = 2
        # A = np.array((2, 2))
        # t = np.array(2)



    # affine + translation
    A = np.zeros((dim, dim))
    t = np.zeros(dim)
    N = len(landmarks)

    params = [np.zeros(dim) for lm in landmarks]
    points = [np.array(lm[0]) for lm in landmarks]

    # B matrix (same for each dimension)
    # B = np.zeros((N + dim + 1, N + dim + 1))  # linear system to solve
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):   # matrix is symmetric w/ 0 diag
            B[i, j] = _sigma(landmarks[i][0], landmarks[j][0], dim)
            B[j, i] = B[i, j]
    # for i in range(N):
    #     B[i, N:N+dim+1] = np.append(landmarks[i][0], 1)
    #     B[N:N+dim+1, i] = B[i, N:N+dim+1]

    # print "final B"
    # print B

    # for each dimension, solve for the parameters separately (lin system: Bx=c)
    # c = np.zeros(N+dim+1)
    c = np.zeros(N)
    for k in range(dim):
        # construct vector c
        for i in range(N):
            c[i] = landmarks[i][1][k]
        # print c

        # Solve linear system
        x = np.linalg.solve(B, c)

        # fill output
        for i in range(N):     # k points
            params[i][k] = x[i]
        # A[k, :] = x[N:N+dim]
        # t[k] = x[-1]

    return params # dict(A=A, t=t, params=params, points=points), B


def circle_and_elipse():

    with torch.no_grad():

        alpha = 1.0
        gamma = 0.001
        step = 0.01

        device = 'cuda:1'

        # Width and height of the black window
        width = 400
        height = 300

        # Create a black window of 400 x 300
        img1 = np.zeros((height, width, 3), np.uint8)
        img2 = np.zeros((height, width, 3), np.uint8)

        # Three vertices(tuples) of the triangle
        p1 = (100, 200)
        p2 = (50, 50)
        p3 = (300, 100)
        pe1 = (188, 155)

        p4 = (100, 100)
        p5 = (150, 50)
        p6 = (300, 200)
        pe2 = (150, 200)
        # Drawing the triangle with the help of lines
        #  on the black window With given points
        # cv2.line is the inbuilt function in opencv library
        cv2.line(img1, p1, p2, (255, 0, 0), 3)
        cv2.line(img1, p2, p3, (255, 0, 0), 3)
        cv2.line(img1, p3, pe1, (255, 0, 0), 3)
        cv2.line(img1, pe1, p1, (255, 0, 0), 3)

        cv2.line(img2, p4, p5, (255, 0, 0), 3)
        cv2.line(img2, p5, p6, (255, 0, 0), 3)
        cv2.line(img2, p6, pe2, (255, 0, 0), 3)
        cv2.line(img2, pe2, p1, (255, 0, 0), 3)

        triangle_cnt1 = np.array([p1, p2, p3, pe1])
        triangle_cnt2 = np.array([p4, p5, p6, pe2])

        cv2.drawContours(img1, [triangle_cnt1], 0, (0, 255, 0), -1)
        cv2.drawContours(img2, [triangle_cnt2], 0, (0, 255, 0), -1)

        triangle_arr1 = img1[:, :, 1]
        triangle_arr2 = img2[:, :, 1]

        # # Create the landmarks
        # source_points = []

        # Flip the landmarks so they are x, y
        source_landmarks = [torch.as_tensor(p) for p in [p4[::-1], p5[::-1], p6[::-1], pe2[::-1]]]
        # source_landmarks = [torch.as_tensor(p) for p in [p4, p5, p6, pe2]]
        source_landmarks = torch.stack(source_landmarks, 0)

        target_landmarks = [torch.as_tensor(p) for p in [p1[::-1], p2[::-1], p3[::-1], pe1[::-1]]]
        # target_landmarks = [torch.as_tensor(p) for p in [p1, p2, p3, pe1]]
        target_landmarks = torch.stack(target_landmarks, 0)

        source_landmarks = source_landmarks.float()
        target_landmarks = target_landmarks.float()

        # # PyCA Stuff
        # landmarks = [
        #     [[p1[1], p1[0]], [p4[1], p4[0]]],
        #     [[p2[1], p2[0]], [p5[1], p5[0]]],
        #     [[p3[1], p3[0]], [p6[1], p6[0]]],
        #     [[pe1[1], pe1[0]], [pe2[1], pe2[0]]]
        # ]
        #
        # # source_landmarks = torch.cat((source_landmarks, torch.ones(len(source_landmarks), 1)), -1).t()
        # # target_landmarks = torch.cat((target_landmarks, torch.ones(len(target_landmarks), 1)), -1).t()
        # #
        #
        #
        # landmarks = [[target_landmarks[x].tolist(), source_landmarks[x].tolist()] for x in range(len(source_landmarks))]
        #
        # pyca_weights = SolveSpline(landmarks)
        #
        # # make an image from the triangle array
        # Img1 = StructuredGrid(
        #     triangle_arr1.shape,
        #     tensor=torch.as_tensor(triangle_arr1, dtype=torch.float32).unsqueeze(0),
        #     origin=[0, 0]
        # )
        # Img2 = StructuredGrid(
        #     triangle_arr2.shape,
        #     tensor=torch.as_tensor(triangle_arr2, dtype=torch.float32).unsqueeze(0),
        #     origin=[0, 0]
        # )
        #
        # ve_filter = VarianceEqualize.Create(kernel_size=11, sigma=3.0, device=device, dtype=torch.float32)
        # test = ve_filter(Img2)
        #
        # source_landmarks = torch.cat((source_landmarks, torch.ones(len(source_landmarks), 1)), -1).t()
        # target_landmarks = torch.cat((target_landmarks, torch.ones(len(target_landmarks), 1)), -1).t()
        # # # Solve for the rigid transform between the points
        # affine = torch.matmul(
        #     torch.matmul(
        #         target_landmarks, source_landmarks.t()
        #     ),
        #     torch.matmul(
        #         source_landmarks, source_landmarks.t()
        #     ).inverse()
        #
        # )
        #
        # # # SVD decomp
        # # U, s, Vt = torch.svd(affine[0:2, 0:2])
        # # affine[0:2, 0:2] = torch.matmul(U, Vt)
        #
        # # Transform the source landmarks
        # affine_landmarks = torch.matmul(affine, source_landmarks)
        #
        # affine_landmarks = affine_landmarks.t()[:, 0:2].contiguous()
        # source_landmarks = source_landmarks.t()[:, 0:2].contiguous()
        # target_landmarks = target_landmarks.t()[:, 0:2].contiguous()
        #
        #
        # transform = AffineTransform.Create(target_landmarks, source_landmarks, rigid=False)
        # img_tformed = transform(Img2)
        #
        # sigma = 0.001
        #
        # # Try using the RB filter
        # rbf_filter_incomp = RadialBasis(target_landmarks, source_landmarks, sigma, incompressible=True, device=device)
        # rbf_filter_regular = RadialBasis(target_landmarks, source_landmarks, sigma, incompressible=False, device=device)
        # rbf_image_incomp = rbf_filter_incomp(Img2)
        # rbf_image_regular = rbf_filter_regular(Img2)
        #
        # reg_vol = torch.abs(Img2.sum() - rbf_image_regular.sum()) / Img2.sum()
        # inc_vol = torch.abs(Img2.sum() - rbf_image_incomp.sum()) / Img2.sum()
        #
        # Display.DispImage(Img1)
        # Display.DispImage(Img2)
        # Display.DispImage(rbf_image_regular)
        # Display.DispImage(rbf_image_incomp)
        #
        # _, camp_param = solve_matrix_spline(target_landmarks, source_landmarks, sigma)
        #
        # # affine = affine.inverse()
        #
        # field = StructuredGrid(triangle_arr1.shape, origin=(0, 0))
        # field *= 0
        #
        # # Create a field that can be used to calculate and incorporate the RBF
        # rbf_temp = StructuredGrid(triangle_arr1.shape, origin=(0, 0))
        # rbf_temp.set_to_identity_lut_()
        #
        # # temp = field.clone()
        #
        # # now apply the fields
        # for i in range(len(target_landmarks)):
        #     # Set the field to identity
        #     rbf_temp.set_to_identity_lut_()
        #     point = target_landmarks[i].view([len(target_landmarks[0])] + len(target_landmarks[0]) * [1]).float()
        #     weight = camp_param[i].view([len(target_landmarks[0])] + len(target_landmarks[0]) * [1]).float()
        #     # Calculate the RBF from the point
        #     rbf_temp.data = -1 * (sigma ** 2) * ((rbf_temp - point) ** 2).data.sum(0)
        #
        #     # rbf_temp.data = rbf_temp.data[0] ** 2 + rbf_temp.data[1] ** 2
        #     # rbf_temp.data[rbf_temp.data < eps] = eps
        #     # rbf_temp.data = (rbf_temp.data ** 2) * torch.log(rbf_temp.data)
        #     rbf_temp.data = torch.exp(rbf_temp.data)
        #     # rbf_temp.data = torch.sqrt(1 + (sigma * rbf_temp.data) ** 2)
        #     # rbf_temp = rbf_temp + point
        #
        #     field.data = field.data + rbf_temp.data * weight
        #     #
        #     # for j in [0, 1]:
        #     #     field.data[j] = field.data[j] + rbf_temp.data[j] * weight[j]
        #
        #     # # Now need to multiply by the weights
        #     # rbf_temp = rbf_temp * weight
        #     #
        #     # # Now add it to the field
        #     # field += rbf_temp
        #
        # # field.data = torch.stack((field.data[1], field.data[0]), 0)
        # rigid_field = StructuredGrid(triangle_arr1.shape, origin=(0, 0))
        # rigid_field.set_to_identity_lut_()
        #
        # affine = affine.inverse()

        # Try making the new affine applier


        # Apply the affine to the field
        # rigid_field.data = torch.matmul(affine[0:2, 0:2].unsqueeze(0).unsqueeze(0), rigid_field.data.permute(1, 2, 0).unsqueeze(-1))
        # rigid_field.data = (rigid_field.data.squeeze() + affine[0:2, 2]).permute(2, 0, 1)
        #
        # rigid_applier = ApplyGrid.Create(rigid_field)
        #
        # img_rigid = rigid_applier(Img2)

        # field.set_identity()
        # applier = ApplyGrid.Create(field)
        #
        # test = applier(img_rigid)
        #
        # Display.DispImage(img_rigid, title='Rigid')
        # Display.DispImage(test, title='Deformed')
        # Display.DispImage(Img1, title='Target')
        # Display.DispImage(Img2, title='Source')

        # Gaussian blur object for the images
        gauss_filt = Gaussian.Create(1, 5, 1, device=device, dim=2)

        # Create the circle image
        circle_im = StructuredGrid((256, 256), device=device)
        circle(circle_im, 20)
        circle_im = gauss_filt(circle_im)

        # Create the ellipse image
        ellipse_im = StructuredGrid((256, 256), device=device)
        ellipse(ellipse_im, 15, 45)
        ellipse_im = gauss_filt(ellipse_im)

        # Load two images that should match
        day0 = LoadITKFile(
            '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/Crop_074_----_3D_VIBE_0.nii.gz',
            device=device,
            dtype=torch.float32
        )
        day0_slice = day0.extract_slice(50, 0)

        # Display.DispImage(day0_slice)

        day3 = LoadITKFile(
            '/home/sci/blakez/ucair/18_047/tpsVolumes/interday/012_-rt-_3D_VIBE_0.5x0.5x1_NoGrappa_3avg_fatsat_cor.nii.gz',
            device=device,
            dtype=torch.float32
        )

        # Need to resample the day3 onto the day0 grid
        day3_resamp = ResampleWorld.Create(day0, device=device)(day3)

        # Display.DispImage(day3_resamp)

        day3_slice = day3_resamp.extract_slice(50, 0)

        # SaveITKFile(day3_resamp, '/home/sci/blakez/18_047_test.nii.gz')

        # Display the images
        # Display.DispImage(ellipse_im, title='Source')
        # Display.DispImage(circle_im, title='Target')

        # Create the smoothing operator
        smoothing = FluidKernel.Create(
            circle_im,
            device=device,
            alpha=alpha,
            beta=0.0,
            gamma=gamma,
            incompresible=False
        )

        # Create the regularization - Does this mean we don't need the smoothing?
        regularization = FluidRegularization.Create(
            circle_im,
            device=device,
            alpha=alpha,
            beta=0.0,
            gamma=gamma,
            incompresible=False
        )

        # Create the matching term
        similarity = L2Similarity(dim=2, device=device)

        # Now make the registration object
        matcher = IterativeMatch.Create(
            source=ellipse_im,
            target=circle_im,
            similarity=similarity,
            regularization=regularization,
            operator=smoothing,
            device=device,
            step_size=step,
            regularization_weight=0.01
        )

        energy = [matcher.initial_energy.item()]
        print(f'Iteration: 0   Energy: {matcher.initial_energy.item()}')
        for i in range(1, 700):

            energy.append(matcher.step().item())

            print(f'Iteration: {i}   Energy: {energy[-1]}')

        # Get the image from the matcher
        def_image = matcher.get_image()
        field = matcher.get_field()
        # Display.DispFieldGrid(field)
        # Display.DispImage(def_image)

        # # Create a jacobian determinant operator
        # jac = JacobianDeterminant.Create(dim=2, device=device)
        # test = jac(field)

        SaveITKFile(def_image, '/home/sci/blakez/18_047_test.nii.gz')

        # Display.DispImage(def_image, title='Moving')

        print('All Done')


if __name__ == '__main__':
    circle_and_elipse()


# def solve_spline(target_landmarks, source_landmarks, sigma=1.0, incomp=False):
#
#     def _sigma(target, source):
#         # r = - sigma * (source ** 2 + target ** 2)
#         # return r.exp()
#         z = (target.float() - source.float()).norm(2)
#         eps = 1e-9
#         return (z ** 2) * torch.log(max(z, eps))
#
#     if target_landmarks.shape != source_landmarks.shape:
#         raise RuntimeError(
#             'Shape of target and source landmarks do not match: '
#             f' Target Shape: {target_landmarks.shape}, Source Shape: {target_landmarks.shape}'
#         )
#
#     shape = source_landmarks.shape
#
#     dim = shape[-1]
#     A = torch.zeros(dim, dim)
#     t = torch.zeros(dim)
#     N = shape[0]
#
#     # Create a matrix of zeros
#     B = torch.zeros((N + dim + 1, N + dim + 1))
#
#     for i in range(N):
#         for j in range(i+1, N):
#             B[i, j] = _sigma(target_landmarks[i], target_landmarks[j])
#             B[j, i] = B[i, j]
#
#     for i in range(N):
#         vec = torch.cat((target_landmarks[i].float(), torch.tensor([1.0])))
#         B[i, N:N + dim + 1] = vec
#         B[N:N + dim + 1, i] = B[i, N:N + dim + 1]
#
#     # Create the RHS vector
#
#     c = torch.zeros(N+dim+1)
#     for k in range(dim):
#
#         for i in range(N):
#             c[i] = source_landmarks[i][k]
#
#         x, _ = torch.solve(c.unsqueeze(-1), B)
#
#     return x, B
