import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(name='libmesh', ext_modules=cythonize("*.pyx"), include_dirs=[numpy.get_include()])
