#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

import subprocess
import os
subprocess.check_call("make")

import numpy


def read_from_makefile(field):
    data = [line for line in open("kernel/Makefile").readlines() if line.startswith(field)][0]
    data = "=" .join(data.split("=")[1:])
    data = data.strip()
    data = data.split(" ")
    return data


objects  = ["kernel/"+obj for obj in read_from_makefile("OBJ")]
extra_compile_args = read_from_makefile("CXXFLAGS")

# replace $(EXTRAFLAGS) in extra_compile_args with environment variable
try:
    del extra_compile_args[extra_compile_args.index("$(EXTRAFLAGS)")]
except ValueError:
    pass
extra_compile_args += os.environ.get('EXTRAFLAGS', "").split()
# extra_compile_args += ["-DCYTHON_TRACE=1"]

kwds = {
    "language": "c++",
    "extra_compile_args": extra_compile_args,
    "extra_link_args": objects,
    "libraries": ["gmp", "pthread"],
    "include_dirs": [numpy.get_include()],
}

extensions = [
    Extension("g6k.siever", ["g6k/siever.pyx"], **kwds),
    Extension("g6k.siever_params", ["g6k/siever_params.pyx"], **kwds)
]


setup(
    name="G6K",
    version="0.0.1",
    ext_modules=cythonize(extensions, compiler_directives={'binding': True,
                                                           'embedsignature': True,
                                                           'language_level': 2}),
    packages=["g6k", "g6k.algorithms", "g6k.utils"],
    package_data={"": ["spherical_coding/*.def"]}
)
