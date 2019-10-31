import torch
import numpy as np
import torch.optim as optim

from FileIO import *
from CAMP.Core import *
from CAMP.UnstructuredGridOperators import *


import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
plt.ion()

device = 'cuda:1'
sigma = 0.5

# Load the surfaces
s1_verts, s1_faces = ReadOBJ('/home/sci/blakez/ucair/test_surfaces/block11_surface_foot.obj')
s2_verts, s2_faces = ReadOBJ('/home/sci/blakez/ucair/test_surfaces/block12_surface_head.obj')

# Create surface objects
foot_surface = TriangleMesh(s1_verts, s1_faces)
head_surface = TriangleMesh(s2_verts, s2_faces)

foot_surface.to_(device)
head_surface.to_(device)

foot_surface.flip_normals_()

# Plot the surfaces
[_, fig, ax] = PlotSurface(foot_surface.vertices, foot_surface.indices)
[src_mesh, _, _] = PlotSurface(head_surface.vertices, head_surface.indices, fig=fig, ax=ax, color=[1, 0, 0])

# Create the affine and translation to be optimized
# Define the parameters to be optimized
init_affine = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().to(device)
affine = init_affine.clone()
affine.requires_grad = True
# translation = torch.mv(affine, block_objects[0].surface_dict['exterior']['mean'].to(device))
translation = (foot_surface.centers.mean(0) - head_surface.centers.mean(0)).clone().float()
translation = translation.to(device)
translation.requires_grad = True

# Create some of the filters
model = AffineCurrents.Create(
    foot_surface.normals, foot_surface.centers, affine, translation, kernel='gaussian', sigma=sigma, device=device
)

# See if we can perform a forward
# loss = model(head_surface.normals.to(device).clone(), head_surface.centers.to(device).clone())

# Create the optimizer
optimizer = optim.SGD([
    {'params': affine, 'lr': 1.0e-06},
    {'params': translation, 'lr': 1.0e-04}], momentum=0.9, nesterov=True
)

for epoch in range(0, 300):
    optimizer.zero_grad()
    loss = model(
        head_surface.normals.clone(), head_surface.centers.clone()
    )

    print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

    loss.backward()  # Compute the gradients
    optimizer.step()  #

    with torch.no_grad():
        U, s, V = affine.clone().svd()
        affine.data = torch.mm(U, V.transpose(1, 0))

    # aff_source_verts = torch.mm(affine,
    #                             (head_surface.vertices - head_surface.centers.mean(0)).permute(1, 0)).permute(1, 0) \
    #                    + translation + head_surface.centers.mean(0)
    # src_mesh.set_verts(aff_source_verts[head_surface.indices].detach().cpu().numpy())
    #
    # plt.draw()
    # plt.pause(0.00001)
#
# # Update the transforms for the source block
affine = affine.detach()
translation = translation.detach()
# # Need to update the translation to account for not rotation about the origin
# translation = translation.cpu().detach()
translation = -torch.matmul(affine, head_surface.centers.mean(0)) + head_surface.centers.mean(0) + translation

full_aff = torch.eye(4)
full_aff[0:3, 0:3] = affine.clone()
full_aff[0:3, 3] = translation.clone().t()


# aff_source_verts = torch.mm(affine,
#                             (head_surface.vertices - head_surface.centers.mean(0)).permute(1, 0)).permute(1, 0) \
#                    + translation + head_surface.centers.mean(0)

# Create affine applier filter and apply
aff_source = AffineTransformSurface.Create(full_aff, device=device)(head_surface)
# aff_source.to_('cpu')
# def update_plot(mesh, verts, faces):
#     verts = apply_affine(verts, affine, translation, mean)
src_mesh.set_verts(aff_source.vertices[aff_source.indices].detach().cpu().numpy())

plt.draw()
plt.pause(0.00001)

print('All Done')