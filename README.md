# CAMP: Computational Anatomy and Medical imaging using PyTorch
![PyPI](https://img.shields.io/pypi/v/camp) ![PyPI - License](https://img.shields.io/pypi/l/camp)

## Documatation
The documentation for camp is avaiable on Read the Docs: [https://sci-camp.readthedocs.io/](https://sci-camp.readthedocs.io/)

## Installation
``pip install camp``

### Requirements
camp depends on these software packages for math routines and data I/O. 

Python 3.6 (or newer) and the following python packages and versions:
- [`numpy`](https://www.numpy.org/) 1.16+
- [`torch`](https://pytorch.org) 1.3+
- [`SimpleITK`](https://simpleitk.org/) 1.2+

## GPU Processing
If available, this code uses a compatible GPU for accelerated computation. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details on how to install pytorch with gpu support for your system. You will then need to install the other dependencies.

### CPU-Only
If you know that you will **not** use a GPU, you can install the CPU-only version of pytorch. See [https://pytorch.org/get-started/locally/#no-cuda-1](https://pytorch.org/get-started/locally/#no-cuda-1) for how to install the CPU-only version. You will then need to install the other dependencies.


## Authors

* **Blake E. Zimmerman** - [Home Page](https://blakezim.github.io/)
* **Markus D. Foote** - [Home Page](https:markusfoote.com)

See also the list of [contributors](https://github.com/blakezim/CAMP/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
