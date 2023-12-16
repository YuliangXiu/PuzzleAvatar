from Cython.Build import cythonize
from setuptools import setup

setup(name='libvoxelize', ext_modules=cythonize("*.pyx"))
