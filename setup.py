#!/usr/bin/env python
# -*- coding: utf-8 -*-
####
#
#   Copyright (C) 2018-2021 Team G6K
#
#   This file is part of G6K. G6K is free software:
#   you can redistribute it and/or modify it under the terms of the
#   GNU General Public License as published by the Free Software Foundation,
#   either version 2 of the License, or (at your option) any later version.
#
#   G6K is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with G6K. If not, see <http://www.gnu.org/licenses/>.
#
####

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

import subprocess
import glob
import numpy

subprocess.check_call("make")


def read_from_pc(field):
    data = [line for line in open("g6k.pc").readlines() if line.startswith(field)][0]
    data = "=" .join(data.split(": ")[1:])
    data = data.strip()
    data = [d for d in data.split(" ") if d]
    return data


objects  = glob.glob("kernel/.libs/*.o")
extra_compile_args = ["-std=c++11"]
extra_compile_args += read_from_pc("Cflags")
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
    description="General Sieve Kernel",
    version="0.1.0",
    url="https://github.com/fplll/g6k",
    ext_modules=cythonize(extensions, compiler_directives={'binding': True,
                                                           'embedsignature': True,
                                                           'language_level': 2}),
    packages=["g6k", "g6k.algorithms", "g6k.utils"],
    package_data={"": ["spherical_coding/*.def"]},
    scripts=["bkz.py",
             "full_sieve.py",
             "hkz.py",
             "hkz_maybe.py",
             "lwe_challenge.py",
             "plain_sieve.py",
             "svp_challenge.py",
             "svp_exact.py",
             "svp_exact_find_norm.py"],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3"
    ]
)
