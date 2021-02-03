.. CAMP documentation master file, created by
   sphinx-quickstart on Sat Dec 19 11:42:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CAMP: Computational Anatomy and Medical Imaging using PyTorch
=============================================================

CAMP is designed as a general-purpose tool for medical image and data processing.
`PyTorch <https://pytorch.org/>`_ provides an efficient backend for mathematical routines (linear algebra, automatic differentiation, etc.) with support for GPU acceleration and distributed computing.
PyTorch also allows CAMP to be portable due to full CPU-only support.

CAMP adopts coordinate system conventions designed to be compatible with medical imaging systems, including meta-data for a consistent world coordinate frame.
Image transformations and image processing can be performed relative to the coordinate system in which the images reside, which facilitates multi-scale algorithms.
Core representations for data include structured and unstructured grids.

Structured Grids
****************
Structured grids represent data with consistent rectilinear element spacing.
These grids are commonly defined by the origin and element spacing attributes.
Images and vector fields are most commonly represented by a structured grid.
CAMP defines many operators to perform computation on structured grid data, including :ref:`Gaussian blur <gaussian-filter>` and :ref:`composition of deformation fields <compose-grids-filter>`.
Multi-channel (including color) images are supported -- the internal convention is channels-first representation (C x D x H x W).

Unstructured Grids
******************
Unstructured grids represent data with arbitrary element shape and location.
Currently, only triangular mesh objects are supported, which aim to represent surface data via edge and vertex representation.
Data values may be face-centered or node-centered.
The unstructured grid objects maintain a world coordinate system that preserves relationships between other unstructured and structured grid data.
An example implementation of deformable surface-based registration is implemented using the unstructured grid representation, based on `Glaunes et al. (2004) <https://ieeexplore.ieee.org/document/1315234?section=abstract>`_.
Watch a summary video using this implementation on `YouTube <https://www.youtube.com/watch?v=RNaI1_TNamY&feature=youtu.be&ab_channel=BlakeZimmerman>`_.

Data I/O
********
Many medical imaging formats are supported through `SimpleITK <https://pypi.org/project/SimpleITK/>`_.


Relevant Publications
=====================
There is not a single publication that describes the architecture or design of the CAMP project. However, the following
publications are use cases that inspired the core development of CAMP.

* **Zimmerman, B. E.**, Johnson, S. L., Odéen, H. A., Shea, J. E., Factor, R. E., Joshi, S. C., & Payne, A. H. (2021). `Histology to 3D In Vivo MR Registration for Volumetric Evaluation of MRgFUS Treatment Assessment Biomarkers <https://arxiv.org/abs/2011.10708)>`_. *Manuscript submitted for publication*.
* **Zimmerman, B. E.**, Johnson, S., Odéen, H., Shea, J., Foote, M. D., Winkler, N., Sarang Joshi, & Payne, A. (2020). `Learning Multiparametric Biomarkers for Assessing MR-Guided Focused Ultrasound Treatments <https://ieeexplore.ieee.org/abstract/document/9200773>`_. IEEE Transactions on Biomedical Engineering.


Table of Contents
=================

.. toctree::
   :maxdepth: 2

   Core
   FileIO
   StructuredGridOperators
   StructuredGridTools
   UnstructuredGridOperators

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
