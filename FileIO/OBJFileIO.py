import torch
import numpy as np


def ReadOBJ(file, device='cpu'):
    with open(file) as f:
        lines = f.readlines()
        verts = np.array([list(map(float, line.split()[1:4])) for line in lines if line.startswith('v ')])
        faces = np.array([list(map(int, line.split()[1:4])) for line in lines if line.startswith('f ')])
        # Subtract 1 because the faces are 1 indexed and need to be 0 indexed for python
        f.close()
        faces -= 1

        verts = torch.tensor(verts, dtype=torch.float, device=device, requires_grad=False)
        faces = torch.tensor(faces, dtype=torch.long, device=device, requires_grad=False)

    return verts, faces


def WriteOBJ(vert, faces, file):
    with open(file, 'w') as f:
        f.write("# OBJ file\n")
        for v in vert.tolist():
            f.write("v")
            for i in range(0, len(v)):
                f.write(" %.4f" % (v[i]))
            f.write("\n")
        for p in faces:
            f.write("f")
            for i in range(0, len(p)):
                f.write(" %d" % (p[i] + 1))
            f.write("\n")
