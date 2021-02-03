# CAMP: Computational Anatomy and Medical imaging using PyTorch
![PyPI](https://img.shields.io/pypi/v/camp) ![PyPI - License](https://img.shields.io/pypi/l/camp)

## Documatation
The documentation for camp is avaiable on Read the Docs: [https://sci-camp.rtfd.io/](https://sci-camp.readthedocs.io/)

## Installation
``pip install camp``

## Overview
CAMP is designed as a general-purpose tool for medical image and data processing.
[PyTorch](https://pytorch.org/) provides an efficient backend for mathematical routines (linear algebra, automatic differentiation, etc.) with support for GPU acceleration and distributed computing.
PyTorch also allows CAMP to be portable due to full CPU-only support.

CAMP adopts coordinate system conventions designed to be compatible with medical imaging systems, including meta-data for a consistent world coordinate frame.
Image transformations and image processing can be performed relative to the coordinate system in which the images reside, which facilitates multi-scale algorithms.
Core representations for data include structured and unstructured grids.
### Structured Grids
Structured grids represent data with consistent rectilinear element spacing.
These grids are commonly defined by the origin and element spacing attributes.
Images and vector fields are most commonly represented by a structured grid.
CAMP defines many operators to perform computation on structured grid data, including Gaussian blur and composition of deformation fields.
Multi-channel (including color) images are supported -- the internal convention is channels-first representation (C x D x H x W).

### Unstructured Grids
Unstructured grids represent data with arbitrary element shape and location.
Currently, only triangular mesh objects are supported, which aim to represent surface data via edge and vertex representation.
Data values may be face-centered or node-centered.
The unstructured grid objects maintain a world coordinate system that preserves relationships between other unstructured and structured grid data.
An example implementation of deformable surface-based registration is implemented using the unstructured grid representation, based on [Glaunes et al. (2004)](https://ieeexplore.ieee.org/document/1315234?section=abstract).
Watch a summary video using this implementation on [YouTube](https://www.youtube.com/watch?v=RNaI1_TNamY&feature=youtu.be&ab_channel=BlakeZimmerman).

### Data I/O
Many medical imaging formats are supported through [SimpleITK](https://pypi.org/project/SimpleITK/).


### Requirements
camp depends on these software packages for math routines and data I/O. 

Python 3.6 (or newer) and the following python packages and versions:
- [`numpy`](https://www.numpy.org/) 1.16+
- [`torch`](https://pytorch.org) 1.3+
- [`SimpleITK`](https://simpleitk.org/) 1.2+

## GPU Processing
If available, this code uses a compatible GPU for accelerated computation. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details on how to install pytorch with gpu support for your system. You will then need to install the other dependencies.

## CPU-Only
If you know that you will **not** use a GPU, you can install the CPU-only version of pytorch. See [https://pytorch.org/get-started/locally/#no-cuda-1](https://pytorch.org/get-started/locally/#no-cuda-1) for how to install the CPU-only version. You will then need to install the other dependencies.


## Authors

* **Blake E. Zimmerman** - [Home Page](https://blakezim.github.io/)
* **Markus D. Foote** - [Home Page](https:markusfoote.com)

See also the list of [contributors](https://github.com/blakezim/CAMP/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Relevant Publications
* **Zimmerman, B. E.**, Johnson, S. L., Odéen, H. A., Shea, J. E., Factor, R. E., Joshi, S. C., & Payne, A. H. (2021). [Histology to 3D In Vivo MR Registration for Volumetric Evaluation of MRgFUS Treatment Assessment Biomarkers.](https://arxiv.org/abs/2011.10708) _Manuscript submitted for publication_. 
* **Zimmerman, B. E.**, Johnson, S., Odéen, H., Shea, J., Foote, M. D., Winkler, N., Sarang Joshi, & Payne, A. (2020). [Learning Multiparametric Biomarkers for Assessing MR-Guided Focused Ultrasound Treatments.](https://ieeexplore.ieee.org/abstract/document/9200773) IEEE Transactions on Biomedical Engineering.