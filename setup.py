from setuptools import setup

setup(
    name='camp',
    version='0.1.1',
    description='CAMP: Computational Anatomy and Medical imaging using PyTorch',
    url='https://github.com/blakezim/CAMP',
    author='Blake E. Zimmerman',
    author_email='blakez@sci.utah.edu',
    license='MIT',
    install_requires=['numpy', 'SimpleITK', 'torch'],
    zip_safe=True
)