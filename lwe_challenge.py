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


"""
LWE Challenge Solving Command Line Client
"""

from __future__ import absolute_import
from __future__ import print_function
import copy
import re
import sys
import time

from collections import OrderedDict # noqa
from math import log

from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from six.moves import range


def lwe_kernel(arg0, params=None, seed=None):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/alpha: the noise rate of the LWE-challenge

        - lwe/m: the number of samples to use for the primal attack

        - lwe/goal_margin: accept anything that is
          goal_margin * estimate(length of embedded vector)
          as an lwe solution

        - lwe/svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the lwe challenge attempt

    """

    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    # params for underlying BKZ
    extra_dim4free = params.pop("bkz/extra_dim4free")
    jump = params.pop("bkz/jump")
    dim4free_fun = params.pop("bkz/dim4free_fun")
    pump_params = pop_prefixed_params("pump", params)
    fpylll_crossover = params.pop("bkz/fpylll_crossover")
    blocksizes = params.pop("bkz/blocksizes")
    tours = params.pop("bkz/tours")

    # flow of the lwe solver
    svp_bkz_time_factor = params.pop("lwe/svp_bkz_time_factor")
    goal_margin = params.pop("lwe/goal_margin")

    # generation of lwe instance and Kannan's embedding
    alpha = params.pop("lwe/alpha")
    m = params.pop("lwe/m")
    decouple = svp_bkz_time_factor > 0

    # misc
    dont_trace = params.pop("dummy_tracer")
    verbose = params.pop("verbose")

    A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    print("-------------------------")
    print("Primal attack, LWE challenge n=%d, alpha=%.4f" % (n, alpha))

    if m is None:
        try:
            min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                        samples=A.nrows, decouple=decouple)
            (b, s, m) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    else:
        try:
            min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q, samples=m,
                                        decouple=decouple)
            (b, s, _) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    print("Chose %d samples. Predict solution at bkz-%d + svp-%d" % (m, b, s))
    print()

    target_norm = goal_margin * (alpha*q)**2 * m + 1

    if blocksizes is not None:
        blocksizes = list(range(10, 40)) + eval("range(%s)" % re.sub(":", ",", blocksizes)) # noqa
    else:
        blocksizes = list(range(10, 50)) + [b-20, b-17] + list(range(b - 14, b + 25, 2))

    B = primal_lattice_basis(A, c, q, m=m)

    g6k = Siever(B, params)
    print("GSO precision: ", g6k.M.float_type)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("lwe"), start_clocks=True)

    d = g6k.full_n
    blocksizes = [blocksize for blocksize in blocksizes if blocksize <= d]
    g6k.lll(0, g6k.full_n)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    T0 = time.time()
    T0_BKZ = time.time()
    for blocksize in blocksizes:
        for tt in range(tours):
            # BKZ tours

            if blocksize < fpylll_crossover:
                if verbose:
                    print("Starting a fpylll BKZ-%d tour. " % (blocksize), end=' ')
                    sys.stdout.flush()
                bkz = BKZReduction(g6k.M)
                par = fplll_bkz.Param(blocksize,
                                      strategies=fplll_bkz.DEFAULT_STRATEGY,
                                      max_loops=1)
                bkz(par)

            else:
                if verbose:
                    print("Starting a pnjBKZ-%d tour. " % (blocksize))

                pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                     verbose=verbose,
                                     extra_dim4free=extra_dim4free,
                                     dim4free_fun=dim4free_fun,
                                     goal_r0=target_norm,
                                     pump_params=pump_params)

            T_BKZ = time.time() - T0_BKZ

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))

            g6k.lll(0, g6k.full_n)

            if g6k.M.get_r(0, 0) <= target_norm:
                break

            # overdoing n_max would allocate too much memory, so we are careful
            svp_Tmax = svp_bkz_time_factor * T_BKZ
            n_max = int(58 + 2.85 * log(svp_Tmax * params.threads)/log(2.))

            rr = [g6k.M.get_r(i, i) for i in range(d)]
            for n_expected in range(2, d-2):
                x = (target_norm/goal_margin) * n_expected/(1.*d)
                if 4./3 * gaussian_heuristic(rr[d-n_expected:]) > x:
                    break

            print("Without otf, would expect solution at pump-%d. n_max=%d in the given time." % (n_expected, n_max)) # noqa
            if n_expected >= n_max - 1:
                continue

            n_max += 1

            # Larger SVP

            llb = d - blocksize
            while gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)]) < target_norm * (d - llb)/(1.*d): # noqa
                llb -= 1
                if llb < 0:
                    break

            # catch small cases where selections above give nonsensical suggestions
            llb = max(0, llb)
            f = max(d-llb-n_max, 0)

            if verbose:
                print("Starting svp pump_{%d, %d, %d}, n_max = %d, Tmax= %.2f sec" % (llb, d-llb, f, n_max, svp_Tmax)) # noqa
            pump(g6k, tracer, llb, d-llb, f, verbose=verbose,
                 goal_r0=target_norm * (d - llb)/(1.*d))

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "\n slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))
                print()

            g6k.lll(0, g6k.full_n)
            T0_BKZ = time.time()
            if g6k.M.get_r(0, 0) <= target_norm:
                break

        if g6k.M.get_r(0, 0) <= target_norm:
            print("Finished! TT=%.2f sec" % (time.time() - T0))
            print(g6k.M.B[0])
            alpha_ = int(alpha*1000)
            filename = 'lwechallenge/%03d-%03d-solution.txt' % (n, alpha_)
            fn = open(filename, "w")
            fn.write(str(g6k.M.B[0]))
            fn.close()
            return

    raise ValueError("No solution found.")


def lwe():
    """
    Attempt to solve an lwe challenge.

    """
    description = lwe.__doc__

    args, all_params = parse_args(description,
                                  lwe__alpha=0.005,
                                  lwe__m=None,
                                  lwe__goal_margin=1.5,
                                  lwe__svp_bkz_time_factor=1,
                                  bkz__blocksizes=None,
                                  bkz__tours=1,
                                  bkz__jump=1,
                                  bkz__extra_dim4free=12,
                                  bkz__fpylll_crossover=51,
                                  bkz__dim4free_fun="default_dim4free_fun",
                                  pump__down_sieve=True,
                                  dummy_tracer=True,  # set to control memory
                                  verbose=True
                                  )

    stats = run_all(lwe_kernel, list(all_params.values()), # noqa
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


if __name__ == '__main__':
    lwe()
