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



from fpylll import IntegerMatrix
try:
    from g6k import Siever
except ImportError:
    raise ImportError("g6k not installed. Please run './setup.py install' from ../")
from simple_pump import pump


n = 80

A = IntegerMatrix.random(n, "qary", k=40, bits=20)
g6k = Siever(A)
print("Squared length of the 5 first GS vectors")

# Do a Workout (pumps of increasing effort), and printout basis shape at each step.
for dim4free in range(50, 15, -1):
    pump(g6k, 0, n, dim4free)
    print("dims4free: %d/%d \t %.3e\t%.3e\t%.3e\t%.3e\t%.3e\t"  %   
          tuple([dim4free, n]+[g6k.M.get_r(i, i) for i in range(5)]))
