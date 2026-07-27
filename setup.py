from __future__ import annotations

import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

import numpy

if sys.platform == "win32":
    composite_compile_args = ["/openmp"]
    composite_link_args = []
else:
    composite_compile_args = ["-fopenmp"]
    composite_link_args = ["-fopenmp"]


extensions = [
    Extension(
        "ditherzam._native._smoke",
        ["ditherzam/_native/_smoke.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "ditherzam._native._composite",
        ["ditherzam/_native/_composite.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=composite_compile_args,
        extra_link_args=composite_link_args,
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3},
    )
)
