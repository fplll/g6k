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


"""

"""
from __future__ import absolute_import
from __future__ import print_function
import sys
from .pump import pump
from fpylll.util import gaussian_heuristic
import time
from six.moves import range


def workout(g6k, tracer, kappa, blocksize, dim4free_min=0,              # Main parameters
            dim4free_dec=1, start_n=40, goal_r0=0.,                     # Loop control
            verbose=False, save_prefix=None, pump_params=None           # Misc
            ):
    """
    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param kappa: beginning of the block
    :param blocksize: dimension of the block
    :param dim4free_min: Minimal number of dimension for free ``dimension for free'' [Ducas,
        Eurcrypt 2018] (may stop before reaching that if goal_r0)
    :param dim4free_dec: By how much do we decreaseee dim4free at each iteration
    :param start_n: Dimension of the first pump
    :param goal_r0: an extra hook to always insert at position kappa if this goal length can be met
        by a lift.  Quit when this is reached.
    :param verbose: Print workout steps (with timing and quality) information on the standard
        output.  Enforce verbosity of pump as well.
    :param save_prefix: If not None, save intermediate basis at a file-name with this prefix.
        Allows to resume computation.
    :param pump_params: Parameters to forward to the pump.

    """
    if pump_params is None:
        pump_params = {}

    f_start = max(blocksize - start_n, 0, dim4free_min)
    fs = list(range(dim4free_min, f_start+1, dim4free_dec))[::-1]

    if goal_r0:
        fs += 9999*[dim4free_min]

    gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(kappa, kappa+blocksize)])
    runtimestart = time.time()

    if "verbose" not in pump_params:
        pump_params["verbose"] = verbose

    with tracer.context(("workout", "beta:%d f:%d" % (blocksize, dim4free_min))):
        for f in fs:
            flast = f
            timestart = time.time()

            sys.stdout.flush()
            pump(g6k, tracer, kappa, blocksize, f, goal_r0=goal_r0, **pump_params)

            if verbose:
                gh2 = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(kappa+f, kappa+blocksize)])
                quality = (gh * (blocksize - f)) / (gh2 * blocksize)
                print("T:%10.5fs, TT:%10.5fs, q:%10.5f r0/gh:%10.5f" %
                      (time.time() - timestart,
                       time.time() - runtimestart, quality, g6k.M.get_r(kappa, kappa) / gh))

            if g6k.M.get_r(kappa, kappa) < goal_r0:
                break

            if save_prefix is not None:
                fn = open("%s_%d_%d.mat" % (save_prefix.rstrip(), g6k.M.d - f, g6k.M.d), "w")
                fn.write(str(g6k.M.B))
                fn.close()

    return flast
