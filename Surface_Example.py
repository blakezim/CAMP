import torch
import numpy as np
import torch.optim as optim

from FileIO import *
from CAMP.Core import *
from CAMP.UnstructuredGridOperators import *


import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt

plt.ion()

device = 'cuda:1'
sigma = 0.5

data_dir = '/hdscratch/ucair/blockface/18_047/surfaces/raw/'

# Blocks to register
tb = 6
sb = 7

# Load the surfaces
tf_verts, tf_faces = ReadOBJ(f'{data_dir}/block{tb:02d}/block{tb:02d}_surface_foot.obj')
sh_verts, sh_faces = ReadOBJ(f'{data_dir}/block{sb:02d}/block{sb:02d}_surface_head.obj')

# Load the other objects for context
se_verts, se_faces = ReadOBJ(f'{data_dir}/block{sb:02d}/block{sb:02d}_surface_exterior.obj')
sf_verts, sf_faces = ReadOBJ(f'{data_dir}/block{sb:02d}/block{sb:02d}_surface_foot.obj')

# Create surface objects
foot_surface = TriangleMesh(tf_verts, tf_faces)
head_surface = TriangleMesh(sh_verts, sh_faces)

exterior_surface = TriangleMesh(se_verts, se_faces)

exterior_surface.add_surface_(sf_verts, sf_faces)

foot_surface.to_(device)
head_surface.to_(device)
exterior_surface.to_(device)

foot_surface.flip_normals_()

# Plot the surfaces
[_, fig, ax] = PlotSurface(foot_surface.vertices, foot_surface.indices)
[src_mesh, _, _] = PlotSurface(head_surface.vertices, head_surface.indices, fig=fig, ax=ax, color=[1, 0, 0])

# Find the inital translation
translation = (foot_surface.centers.mean(0) - head_surface.centers.mean(0)).clone()

# Create some of the filters
model = AffineCurrents.Create(
    foot_surface.normals,
    foot_surface.centers,
    sigma=sigma,
    init_translation=translation,
    kernel='cauchy',
    device=device
)

# Create the optimizer
optimizer = optim.SGD([
    {'params': model.affine, 'lr': 1.0e-06},
    {'params': model.translation, 'lr': 1.0e-04}], momentum=0.9, nesterov=True
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
        U, s, V = model.affine.clone().svd()
        model.affine.data = torch.mm(U, V.transpose(1, 0))

    # aff_source_verts = torch.mm(affine,
    #                             (head_surface.vertices - head_surface.centers.mean(0)).permute(1, 0)).permute(1, 0) \
    #                    + translation + head_surface.centers.mean(0)
    # src_mesh.set_verts(aff_source_verts[head_surface.indices].detach().cpu().numpy())
    #
    # plt.draw()
    # plt.pause(0.00001)
#
# # Update the transforms for the source block
affine = model.affine.detach()
translation = translation.detach()
# # Need to update the translation to account for not rotation about the origin
translation = -torch.matmul(affine, head_surface.centers.mean(0)) + head_surface.centers.mean(0) + translation

full_aff = torch.eye(4)
full_aff[0:3, 0:3] = affine.clone()
full_aff[0:3, 3] = translation.clone().t()

# Create affine applier filter and apply
aff_tfrom = AffineTransformSurface.Create(full_aff, device=device)
aff_source_head = aff_tfrom(head_surface)
aff_source_exterior = aff_tfrom(exterior_surface)
# aff_source.to_('cpu')
# def update_plot(mesh, verts, faces):
#     verts = apply_affine(verts, affine, translation, mean)
src_mesh.set_verts(aff_source_head.vertices[aff_source_head.indices].detach().cpu().numpy())

plt.draw()
plt.pause(0.00001)

### Now do deformable registration

# Create the deformable model
model = DeformableCurrents.Create(
    aff_source_head.copy(), foot_surface, sigma=sigma, kernel='cauchy', device=device
)

# Create a smoothing filter
sigma = torch.tensor([0.5, 0.5, 5.0], device=device)
gauss = GaussianSmoothing(sigma, dim=3, device=device)

# Set up the optimizer
optimizer = optim.SGD([
    {'params': [model.src_vertices, aff_source_exterior.vertices], 'lr': 5.0e-05}], momentum=0.9, nesterov=True
)

# Create a structured grid for PHI inverse
# test = StructuredGrid((128, 128, 30), device=device, dtype=torch.float32, requires_grad=False)

# Now iterate
for epoch in range(0, 1500):
    optimizer.zero_grad()
    loss = model()

    print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

    loss.backward()  # Compute the gradients

    # Need to propegate the gradients to the other vertices
    with torch.no_grad():
        d = ((aff_source_exterior.vertices.unsqueeze(1) - model.src_vertices.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
        gauss_d = torch.exp(-d / (2 * sigma[None, None, :]))
        aff_source_exterior.vertices.grad = (model.src_vertices.grad[None, :, :].repeat(len(d), 1, 1) * gauss_d).sum(1)

    # Now the gradients are stored in the parameters being optimized
    model.src_vertices.grad = gauss(model.src_vertices)
    optimizer.step()  #

# Update the plot
src_mesh.set_verts(model.src_vertices[model.src_indices].detach().cpu().numpy())

plt.draw()
plt.pause(0.00001)

# Add the surfaces together
aff_source_exterior.add_surface_(model.src_vertices, model.src_indices)
# exterior_surface.add_surface_(sf_verts, sf_faces)

WriteOBJ(aff_source_exterior.vertices, aff_source_exterior.indices, f'/home/sci/blakez/block07_exterior_def.obj')
# WriteOBJ(model.src_vertices, model.src_indices, f'/home/sci/blakez/block08_reg_face_def.obj')

print('All Done')