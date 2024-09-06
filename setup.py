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
    from setuptools.command import build_ext as build_module
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command import build_ext as build_module
from Cython.Build import cythonize

import subprocess
import numpy
import sys
import os
from ast import parse

#
# `setup,py` consumes files output by `configure` so we insist on running it first.
#

if not os.path.exists("configure"):
    subprocess.check_call(["autoreconf", "-i"])

if not os.path.exists("Makefile"):
    subprocess.check_call("./configure")


#
# But we only run `make` as part of `build_ext`
#


class build_ext(build_module.build_ext):
    def run(self):

        for arg in sys.argv:
            if arg.startswith("-j"):
                subprocess.check_call(["make", arg])
                break
        else:
            subprocess.check_call("make")

        build_module.build_ext.run(self)


def read_from(filename, field, sep):
    data = [line for line in open(filename).readlines() if line.startswith(field)][0]
    data = "=".join(data.split(sep)[1:])
    data = data.strip()
    data = [d for d in data.split(" ") if d]
    return data


# Version

with open(os.path.join("g6k", "__init__.py")) as f:
    __version__ = (
        parse(next(filter(lambda line: line.startswith("__version__"), f))).body[0].value.s
    )

extra_compile_args = ["-std=c++11"]
# extra_compile_args += ["-DCYTHON_TRACE=1"]
# there's so many warnings generated here, we need to filter out -Werror
extra_compile_args += [opt for opt in read_from("g6k.pc", "Cflags", ": ") if opt != "-Werror"]

kwds = {
    "language": "c++",
    "extra_compile_args": extra_compile_args,
    # TODO: we should just install the shared lib and link against that
    "extra_link_args": [
        ("kernel/.libs/%s" % fn).replace(".cpp", ".o")
        for fn in read_from("kernel/Makefile.am", "libg6k_la_SOURCES", "=")
    ],
    "libraries": ["gmp", "pthread"],
    "include_dirs": [numpy.get_include()],
}

extensions = [
    Extension("g6k.siever", ["g6k/siever.pyx"], **kwds),
    Extension("g6k.siever_params", ["g6k/siever_params.pyx"], **kwds),
    Extension("g6k.slicer", ["g6k/slicer.pyx"], **kwds),
]

setup(
    name="g6k",
    description="General Sieve Kernel",
    version=__version__,
    url="https://github.com/fplll/g6k",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"binding": True, "embedsignature": True, "language_level": 2},
    ),
    packages=["g6k", "g6k.algorithms", "g6k.utils"],
    package_data={"": ["spherical_coding/*.def"]},
    cmdclass={
        "build_ext": build_ext,
    },
    author="G6K team",
    author_email="fplll-devel@googlegroups.com",
    scripts=[
        "bkz.py",
        "full_sieve.py",
        "hkz.py",
        "hkz_maybe.py",
        "lwe_challenge.py",
        "plain_sieve.py",
        "svp_challenge.py",
        "svp_exact.py",
        "svp_exact_find_norm.py",
    ],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
)
