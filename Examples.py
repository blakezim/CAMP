import torch
import cv2
import numpy as np
from Core import *
from ImageOperators import *
from ImageTools import *
from FileIO import *
import glob

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
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
        device = 'cuda:1'

        # # # landmarks = [[[-5.067165, 46.25, 9.768159], [-7.708513, 45.529398, 9.724182]], [[-22.625597, 40.75, -18.203844], [-26.124477, 39.029398, -16.200991]], [[-24.201354, 50.25, 6.560102], [-27.921156, 49.529398, 7.670703]], [[8.319126, 55.25, 48.231431], [3.007396, 52.529398, 43.927441]], [[2.621867, 62.75, 4.656731], [0.055709, 62.029398, 2.857861]], [[11.620947, 74.75, 40.267556], [11.734125, 74.529398, 36.611922]], [[-41.48797, 72.590742, 21.699997], [-46.958948, 72.661284, 23.026169]], [[-37.075866, 59.26558, 7.199997], [-40.845577, 58.070459, 8.526169]], [[-8.416497, 67.191336, 12.199997], [-10.895197, 66.033761, 14.026169]], [[35.410681, 75.057844, -1.800003], [36.612227, 75.016824, -1.473831]], [[13.5, 65.377428, -3.869874], [12.125687, 63.77973, -3.601394]], [[-10.0, 56.069879, -12.792643], [-12.374313, 54.358263, -11.42074]], [[22.0, 52.655275, -14.445007], [20.125687, 52.124307, -14.674753]], [[32.0, 72.482007, -12.572327], [37.125687, 74.269611, -12.343519]]]
        rigid_matrix = [[0.9689578463919097, -0.006562321241736855, -0.2471388837384192, -4.528427048834612], [0.045088641005423334, 0.9875727458506949, 0.15055592351415986, -12.121896486318946], [0.24307962968507982, -0.15702549981718802, 0.9572122471214657, 30.248595664088207], [0.0, 0.0, 0.0, 1.0]]
        rigid_tensor = torch.as_tensor(rigid_matrix)
        rigid_tensor = rigid_tensor.to(device)
        rigid_tensor = rigid_tensor.type(torch.float32)

        source_volume = LoadITKFile(
            '/home/sci/blakez/ucair/18_047/rawVolumes/PostImaging_2018-07-02/012_----_3D_VIBE_0.5x0.5x1_NoGrappa_3avg_fatsat_cor.nii.gz',
            device=device
        )

        target_volume = LoadITKFile(
            '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/074_----_3D_VIBE_0.5x0.5x1_NoGrappa_3avg_fatsat_cor_post.nii.gz',
            device=device
        )

        target_list = sorted(glob.glob('/home/sci/blakez/ucair/18_047/landmarks/day0_tps/*.txt'))
        source_list = sorted(glob.glob('/home/sci/blakez/ucair/18_047/landmarks/day3_tps/*.txt'))

        for src_item, tar_item in zip(source_list, target_list):
            rigid_array = np.loadtxt(src_item)
            target_array = np.loadtxt(tar_item)

            # rigid_array = rigid_array[:, ::-1].copy() # Can't do this yet because the transform expects xyz
            target_array = target_array[:, ::-1].copy()

            rigid_landmarks = torch.as_tensor(rigid_array)
            target_landmarks = torch.as_tensor(target_array)

            rigid_landmarks = rigid_landmarks.to(device)
            rigid_landmarks = rigid_landmarks.type(torch.float32)
            target_landmarks = target_landmarks.to(device)
            target_landmarks = target_landmarks.type(torch.float32)

        # Need to rotate the point by the (inverse?) of the rigid matrix
        rigid_landmarks = torch.cat((rigid_landmarks, torch.ones(len(rigid_landmarks), 1, device=device)), -1).t()
        source_landmarks = torch.matmul(rigid_tensor.inverse(), rigid_landmarks)

        source_landmarks = source_landmarks.t()[:, 0:3].contiguous()
        rigid_landmarks = rigid_landmarks.t()[:, 0:3].contiguous()

        source_array = source_landmarks.cpu().numpy()
        source_array = source_array[:, ::-1].copy()

        source_landmarks = torch.as_tensor(source_array)
        source_landmarks = source_landmarks.to(device)
        source_landmarks = source_landmarks.type(torch.float32)

        sigma = 0.9

        rbf_filter = RadialBasis.Create(target_landmarks, source_landmarks, sigma, incompressible=True, device=device)
        test = rbf_filter(source_volume, target_volume)
        SaveITKFile(test, '/home/sci/blakez/no_chance_incomp.nii.gz')
        jacobian = JacobianDeterminant.Create(dim=3, device=device, dtype=torch.float32)
        # print('Done')
        # # Width and height of the black window
        # width = 400
        # height = 300
        #
        # # Create a black window of 400 x 300
        # img1 = np.zeros((height, width, 3), np.uint8)
        # img2 = np.zeros((height, width, 3), np.uint8)
        #
        # # Three vertices(tuples) of the triangle
        # p1 = (100, 200)
        # p2 = (50, 50)
        # p3 = (300, 100)
        # pe1 = (188, 155)
        #
        # p4 = (100, 100)
        # p5 = (150, 50)
        # p6 = (300, 200)
        # pe2 = (150, 200)
        # # Drawing the triangle with the help of lines
        # #  on the black window With given points
        # # cv2.line is the inbuilt function in opencv library
        # cv2.line(img1, p1, p2, (255, 0, 0), 3)
        # cv2.line(img1, p2, p3, (255, 0, 0), 3)
        # cv2.line(img1, p3, pe1, (255, 0, 0), 3)
        # cv2.line(img1, pe1, p1, (255, 0, 0), 3)
        #
        # cv2.line(img2, p4, p5, (255, 0, 0), 3)
        # cv2.line(img2, p5, p6, (255, 0, 0), 3)
        # cv2.line(img2, p6, pe2, (255, 0, 0), 3)
        # cv2.line(img2, pe2, p1, (255, 0, 0), 3)
        #
        # triangle_cnt1 = np.array([p1, p2, p3, pe1])
        # triangle_cnt2 = np.array([p4, p5, p6, pe2])
        #
        # cv2.drawContours(img1, [triangle_cnt1], 0, (0, 255, 0), -1)
        # cv2.drawContours(img2, [triangle_cnt2], 0, (0, 255, 0), -1)
        #
        # triangle_arr1 = img1[:, :, 1]
        # triangle_arr2 = img2[:, :, 1]
        #
        # # # Create the landmarks
        # # source_points = []
        #
        # # Flip the landmarks so they are x, y
        # source_landmarks = [torch.as_tensor(p) for p in [p4[::-1], p5[::-1], p6[::-1], pe2[::-1]]]
        # # source_landmarks = [torch.as_tensor(p) for p in [p4, p5, p6, pe2]]
        # source_landmarks = torch.stack(source_landmarks, 0)
        #
        # target_landmarks = [torch.as_tensor(p) for p in [p1[::-1], p2[::-1], p3[::-1], pe1[::-1]]]
        # # target_landmarks = [torch.as_tensor(p) for p in [p1, p2, p3, pe1]]
        # target_landmarks = torch.stack(target_landmarks, 0)
        #
        # source_landmarks = source_landmarks.float()
        # target_landmarks = target_landmarks.float()

        # PyCA Stuff
        # landmarks = [
        #     [[p1[1], p1[0]], [p4[1], p4[0]]],
        #     [[p2[1], p2[0]], [p5[1], p5[0]]],
        #     [[p3[1], p3[0]], [p6[1], p6[0]]],
        #     [[pe1[1], pe1[0]], [pe2[1], pe2[0]]]
        # ]

        # source_landmarks = torch.cat((source_landmarks, torch.ones(len(source_landmarks), 1)), -1).t()
        # target_landmarks = torch.cat((target_landmarks, torch.ones(len(target_landmarks), 1)), -1).t()
        #


        # landmarks = [[target_landmarks[x].tolist(), source_landmarks[x].tolist()] for x in range(len(source_landmarks))]

        # pyca_weights = SolveSpline(landmarks)

        # make an image from the triangle array
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
        # sigma = 0.5
        #
        # Img2.to_(device)
        # Img2.to_type_(torch.float32)
        #
        # source_landmarks = source_landmarks.to(device)
        # source_landmarks = source_landmarks.type(torch.float32)
        #
        # target_landmarks = target_landmarks.to(device)
        # target_landmarks = target_landmarks.type(torch.float32)
        # #
        # # # Try using the RB filter
        # rbf_filter_incomp = RadialBasis.Create(target_landmarks,
        #                                        source_landmarks, sigma=[2.0, 2.0], incompressible=True, device=device)
        # rbf_filter_regular = RadialBasis.Create(target_landmarks,
        #                                         source_landmarks, sigma=1.0, incompressible=False, device=device)
        # rbf_image_incomp = rbf_filter_incomp(Img2)
        # rbf_image_regular = rbf_filter_regular(Img2)
        # #
        # # reg_vol = torch.abs(Img2.sum() - rbf_image_regular.sum()) / Img2.sum()
        # # inc_vol = torch.abs(Img2.sum() - rbf_image_incomp.sum()) / Img2.sum()
        # #
        # Display.DispImage(Img1)
        # Display.DispImage(Img2)
        # Display.DispImage(rbf_image_regular)
        # Display.DispImage(rbf_image_incomp)
        # print('Done')
        # #
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
        # # field.data = torch.stack((field.data.0[1], field.data[0]), 0)
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

        step = 0.000001

        # Gaussian blur object for the images
        gauss_filt = Gaussian.Create(1, 5, 1, device=device, dim=2)

        # Create the circle image
        circle_im = StructuredGrid((130, 200), device=device)
        circle(circle_im, 40)
        circle_im = gauss_filt(circle_im)

        # Create the ellipse image
        ellipse_im = StructuredGrid((130, 200), device=device)
        ellipse(ellipse_im, 35, 75)
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
            incompresible=True
        )

        # Create the regularization - Does this mean we don't need the smoothing?
        regularization = FluidRegularization.Create(
            circle_im,
            device=device,
            alpha=alpha,
            beta=0.0,
            gamma=gamma,
            incompresible=True
        )

        # Create the matching term
        # similarity = L2Similarity(dim=2, device=device)
        similarity = NormalizedCrossCorrelation.Create(grid=circle_im, device=device)

        # Now make the registration object
        matcher_incomp = IterativeMatch.Create(
            source=ellipse_im,
            target=circle_im,
            similarity=similarity,
            regularization=regularization,
            operator=smoothing,
            device=device,
            step_size=step,
            regularization_weight=0.1
        )

        energy_incomp = [matcher_incomp.initial_energy.item()]
        print(f'Iteration: 0   Energy: {matcher_incomp.initial_energy.item()}')
        for i in range(1, 500):

            energy_incomp.append(matcher_incomp.step().item())

            print(f'Iteration: {i}   Energy: {energy_incomp[-1]}')

        # Get the image from the matcher
        def_image_incomp = matcher_incomp.get_image()
        field_incomp = matcher_incomp.get_field()

        # Now do the same thing for the compressible version
        regularization.fluid_operator.incompresible = False
        smoothing.incompresible = False

        matcher_comp = IterativeMatch.Create(
            source=ellipse_im,
            target=circle_im,
            similarity=similarity,
            regularization=regularization,
            operator=smoothing,
            device=device,
            step_size=step,
            regularization_weight=0.1
        )

        energy_comp = [matcher_comp.initial_energy.item()]
        print(f'Iteration: 0   Energy: {matcher_comp.initial_energy.item()}')
        for i in range(1, 500):
            energy_comp.append(matcher_incomp.step().item())

            print(f'Iteration: {i}   Energy: {energy_comp[-1]}')

        def_image_comp = matcher_incomp.get_image()
        field_comp = matcher_incomp.get_field()

        jacobain = JacobianDeterminant.Create(dim=2, device=device)
        jacobian_comp = jacobain(field_comp)
        jacobian_incomp = jacobain(field_incomp)

        jacobian_comp.data[jacobian_comp.data <0.001] = 0.01

        Display.DispImage(def_image_comp, title='Moving Compressible')
        Display.DispImage(def_image_incomp, title='Moving Incompressible')
        Display.DispImage(ellipse_im, title='Source')
        Display.DispImage(circle_im, title='Target')
        Display.DispImage(jacobian_comp, title='Jacobian Determinant Compressible', rng=[0.5, 6], cmap='jet')
        Display.DispImage(jacobian_incomp, title='Jacobian Determinant Compressible', rng=[0.5, 6], cmap='jet')
        Display.DispFieldGrid(field_comp, title='Deformation Grid Compressible')
        Display.DispFieldGrid(field_incomp, title='Deformation Grid Incompressible')
        # Display.EnergyPlot(energy_comp, title='Compressible Energy')
        Display.EnergyPlot([energy_comp, energy_incomp], title='Energy Plots', legend=['Compressible', 'Incompressible'])
        Display.DispImage(circle_im - ellipse_im, title='Difference Image')

        font = {'family': 'sans-serif',
                'size': 11}

        matplotlib.rc('font', **font)

        fig = plt.figure()
        gs_master = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.1)
        gs00 = gs_master[0, 0].subgridspec(1, 3, width_ratios=[2.7, 1, 2.7], wspace=0.05, hspace=0.2)
        gs10 = gs_master[1, 0].subgridspec(2, 3, wspace=0.05, hspace=0.2)

        # gs = gridspec.GridSpec(3, 3)
        # inner = gridspec.GridSpecFromSubplotSpec(1, 3, gs[0:3, 0])
        # gs.update(wspace=0.05)
        ax00 = plt.subplot(gs00[0, 0])
        ax01 = plt.subplot(gs00[0, 1])
        ax02 = plt.subplot(gs00[0, 2])
        ax10 = plt.subplot(gs10[0, 0])
        ax11 = plt.subplot(gs10[0, 1])
        ax12 = plt.subplot(gs10[0, 2])
        ax20 = plt.subplot(gs10[1, 0])
        ax21 = plt.subplot(gs10[1, 1])
        ax22 = plt.subplot(gs10[1, 2])

        plt.sca(ax00)
        ax00.set_anchor('E')
        plt.imshow(ellipse_im.data.squeeze().cpu(), cmap='gray')
        ax00.text(5, 125, f'V={ellipse_im.sum().item():.1f}', **font, color='w')
        plt.title('Moving Image', **font)
        plt.axis('off')

        plt.sca(ax01)
        ax01.text(0.4, 0.5, '$\phi^{-1}$', family='sans-serif', size=30)
        plt.arrow(0.2, 0.4, 0.6, 0.0, width=.01, color='k')
        plt.axis('off')

        plt.sca(ax02)
        ax02.set_anchor('W')
        plt.imshow(circle_im.data.squeeze().cpu(), cmap='gray')
        plt.title('Target Image', **font)
        plt.axis('off')

        plt.sca(ax10)
        ax10.set_anchor('E')
        plt.imshow(def_image_comp.data.squeeze().cpu(), cmap='gray')
        plt.title('Non-Volume Preserving Deformed Image', **font)
        change = (ellipse_im.sum().item() - def_image_comp.sum().item()) / ellipse_im.sum().item() * 100
        ax10.text(5, 125, f'dV={change:.1f}%', **font, color='w')
        plt.axis('off')

        # START Make the field arrays
        field = field_comp.data.clone()
        field = field - field_comp.origin.view(*field_comp.size.shape, *([1] * len(field_comp.size)))
        field = field / (field_comp.spacing * (field_comp.size / 2)).view(*field_comp.size.shape, *([1] * len(field_comp.size)))
        field = field - 1
        field_y = field[-1].cpu().detach().squeeze().numpy()  # X Coordinates
        field_x = field[-2].cpu().detach().squeeze().numpy()  # Y Coordinates
        sy = field_comp.size[-1].item()
        sx = field_comp.size[-2].item()
        grid_sizex = max(sx // 64, 1)
        grid_sizey = max(sy // 64, 1)
        grid_sizex = int(grid_sizex)
        grid_sizey = int(grid_sizey)
        hx_sample_h = field_x[grid_sizex // 2::grid_sizex, :]
        hy_sample_h = field_y[grid_sizex // 2::grid_sizex, :]
        hx_sample_v = field_x[:, grid_sizey // 2::grid_sizey]
        hy_sample_v = field_y[:, grid_sizey // 2::grid_sizey]
        minax = -1.0
        maxax = 1.0

        # STOP Make the field arrays

        plt.sca(ax11)
        plt.axis([minax, maxax, maxax, minax])
        plt.plot(hy_sample_h.transpose(), hx_sample_h.transpose(), 'k')
        plt.plot(hy_sample_v, hx_sample_v, 'k')
        plt.axis('off')
        plt.title('$\phi^{-1}$ Non-Volume Preserving Deformation Field', **font)

        plt.sca(ax12)
        ax12.set_anchor('W')
        plt.title('Non-Volume Preserving Jacobian Determinant', **font)
        plt.axis('off')
        img12 = plt.imshow((jacobian_comp).data.squeeze().cpu(), cmap='jet', vmin=0.5, vmax=6.0)
        divider = make_axes_locatable(ax12)
        cax12 = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(img12, cax=cax12)
        divider.set_anchor('W')

        plt.sca(ax20)
        ax20.set_anchor('E')
        plt.imshow(def_image_incomp.data.squeeze().cpu(), cmap='gray')
        plt.title('Volume Preserving Deformed Image', **font)
        change = (ellipse_im.sum().item() - def_image_incomp.sum().item()) / ellipse_im.sum().item() * 100
        ax20.text(5, 125, f'dV={change:.1f}%', **font, color='w')
        plt.axis('off')

        # START Make the field arrays
        field = field_incomp.data.clone()
        field = field - field_incomp.origin.view(*field_incomp.size.shape, *([1] * len(field_incomp.size)))
        field = field / (field_incomp.spacing * (field_incomp.size / 2)).view(*field_comp.size.shape,
                                                                              *([1] * len(field_incomp.size)))
        field = field - 1
        field_y = field[-1].cpu().detach().squeeze().numpy()  # X Coordinates
        field_x = field[-2].cpu().detach().squeeze().numpy()  # Y Coordinates
        sy = field_incomp.size[-1].item()
        sx = field_incomp.size[-2].item()
        grid_sizex = max(sx // 64, 1)
        grid_sizey = max(sy // 64, 1)
        grid_sizex = int(grid_sizex)
        grid_sizey = int(grid_sizey)
        hx_sample_h = field_x[grid_sizex // 2::grid_sizex, :]
        hy_sample_h = field_y[grid_sizex // 2::grid_sizex, :]
        hx_sample_v = field_x[:, grid_sizey // 2::grid_sizey]
        hy_sample_v = field_y[:, grid_sizey // 2::grid_sizey]
        minax = -1.0
        maxax = 1.0

        # STOP Make the field arrays

        plt.sca(ax21)
        plt.axis([minax, maxax, maxax, minax])
        plt.plot(hy_sample_h.transpose(), hx_sample_h.transpose(), 'k')
        plt.plot(hy_sample_v, hx_sample_v, 'k')
        plt.axis('off')
        plt.title('$\phi^{-1}$ Volume-Preserving Deformation Field', **font)

        plt.sca(ax22)
        ax22.set_anchor('W')
        plt.title('Volume Preserving Jacobian Determinant', **font)
        plt.axis('off')
        img22 = plt.imshow((jacobian_incomp).data.squeeze().cpu(), cmap='jet', vmin=0.5, vmax=6.0)
        divider = make_axes_locatable(ax22)
        cax22 = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(img22, cax=cax22)
        divider.set_anchor('W')

        fig.set_size_inches(18, 11)
        fig.set_dpi(100)

        plt.savefig('/home/sci/blakez/Volume_preserving_registration_ex.png', dpi=500)
        # plt.savefig('/home/sci/blakez/Volume_preserving_registration_ex.eps', dpi=500)
        # plt.savefig('/home/sci/blakez/Volume_preserving_registration_ex.pdf', dpi=500)

        print('Done')

        # Create an image to show the difference between incompressible and compressible

        # # Create a Jacobian determinant operator
        # jacobain = JacobianDeterminant.Create(dim=2, device=device)
        # jacobian_image = jacobain(field)
        #
        # Display.DispImage(circle_im, title='Target')
        # Display.DispImage(ellipse_im, title='Source')
        # Display.DispImage(def_image, title='Moving')
        # Display.DispFieldGrid(field, title='Deformation')
        # Display.EnergyPlot(energy, title='Energy')
        # Display.DispImage(jacobian_image, title='Jacobian Determinant')
        #
        # SaveITKFile(def_image, '/home/sci/blakez/test.nii.gz')


if __name__ == '__main__':
    circle_and_elipse()
