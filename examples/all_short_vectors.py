#!/usr/bin/env python
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



# SCRIPT: Get (almost) all short vectors of squared length < 1.7 * gh^2

from math import sqrt
from fpylll import IntegerMatrix
from fpylll.util import gaussian_heuristic

try:
    from g6k import Siever
except ImportError:
    raise ImportError("g6k not installed. Please run './setup.py install' from ../")

from simple_pump import pump

# Set up the instance

n = 30
A = IntegerMatrix.random(n, "qary", k=n/2, bits=8)
g6k = Siever(A)
g6k.lll(0, n)

# Run a progressive sieve, to get an initial database saturating the 4/3 *gh^2 ball.
# This makes the final computation saturating a larger ball faster than directly
# running such a large sieve.
#
# G6K will use a database of size db_size_base * db_size_base^dim with default values 
#               db_size_base = sqrt(4./3) and db_size_factor = 3.2
#
# The sieve stops when a 
#               0.5 * saturation_ratio * saturation_radius^(dim/2) 
#
# distinct vectors (modulo negation) of length less than saturation_radius have
# been found

g6k.initialize_local(0, n/2, n)
while g6k.l > 0:
    # Extend the lift context to the left
    g6k.extend_left(1)
    # Sieve
    g6k()


# Increase db_size, saturation radius and saturation ratio to find almost all 
# the desired vectors.
# We need to increase db_size to at least 
#               0.5 * saturation_ratio * saturation_radius^(dim/2)
#
# for this to be possible, but a significant margin is recommended

with g6k.temp_params(saturation_ratio=.95, saturation_radius=1.7, 
                     db_size_base=sqrt(1.7), db_size_factor=5):
    g6k()

# Convert all db vectors from basis A to cannonical basis and print them 
# out if they are indeed shorter than 1.7 * gh^2

gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(n)])

db = list(g6k.itervalues())
found = 0

for x in db:
    v = A.multiply_left(x)
    l = sum(v_**2 for v_ in v)
    if l < 1.7 * gh:
        print(l/gh, v)
        found += 1

print("found %d vectors of squared length than 1.7*gh. (expected %f)"%(found, .5 * 1.7**(n/2.)))
