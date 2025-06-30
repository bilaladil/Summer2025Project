try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy


setup(name = 'Utils_g2pp',
      ext_modules = cythonize("utils_g2pp_newton.pyx"),
	  include_dirs=[numpy.get_include()]
)
