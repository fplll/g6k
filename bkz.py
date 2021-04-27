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
BKZ Command Line Client
"""

from __future__ import absolute_import
from __future__ import print_function
import logging
import re
import time
import pickle as pickler

from collections import OrderedDict

from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix
from fpylll.tools.quality import basis_quality

from g6k.algorithms.bkz import naive_bkz_tour, pump_n_jump_bkz_tour, slide_tour
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from g6k.utils.util import load_prebkz
from g6k.utils.util import sanitize_params_names, print_stats, output_profiles, db_stats
import six
import numpy as np
from six.moves import range


def bkz_kernel(arg0, params=None, seed=None):
    """
    Run the BKZ algorithm with different parameters.

    :param d: the dimension of the lattices to BKZ reduce
    :param params: parameters for BKZ:

        - bkz/alg: choose the underlying BKZ from
          {fpylll, naive, pump_n_jump, slide}

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/pre_blocksize: prereduce lattice with fpylll BKZ up
          to this blocksize

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump
        
        - slide/overlap: shift of the dual blocks when running slide reduction

        - challenge_seed: a seed to randomise the generated lattice

        - dummy_tracer: use a dummy tracer which capture less information

        - verbose: print tracer information throughout BKZ run

    """
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        d, params, seed = arg0
    else:
        d = arg0

    # params for underlying BKZ/workout/pump
    dim4free_fun = params.pop("bkz/dim4free_fun")
    extra_dim4free = params.pop("bkz/extra_dim4free")
    jump = params.pop("bkz/jump")
    overlap = params.pop("slide/overlap")
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)

    # flow of the bkz experiment
    algbkz = params.pop("bkz/alg")
    blocksizes = params.pop("bkz/blocksizes")
    blocksizes = eval("range(%s)" % re.sub(":", ",", blocksizes))
    pre_blocksize = params.pop("bkz/pre_blocksize")
    tours = params.pop("bkz/tours")

    # misc
    verbose = params.pop("verbose")
    dont_trace = params.pop("dummy_tracer", False)

    if blocksizes[-1] > d:
        print('set a smaller maximum blocksize with --blocksizes')
        return

    challenge_seed = params.pop("challenge_seed")

    A, bkz = load_prebkz(d, s=challenge_seed, blocksize=pre_blocksize)

    MM = GSO.Mat(A, float_type="double",
                 U=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                 UinvT=IntegerMatrix.identity(A.nrows, int_type=A.int_type))

    g6k = Siever(MM, params, seed=seed)
    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("bkz", d), start_clocks=True)

    if algbkz == "fpylll":
        M = bkz.M
    else:
        M = g6k.M

    T0 = time.time()
    for blocksize in blocksizes:

        for t in range(tours):
            with tracer.context("tour", t, dump_gso=True):
                if algbkz == "fpylll":
                    par = BKZ_FPYLLL.Param(blocksize,
                                           strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
                                           max_loops=1)
                    bkz(par)

                elif algbkz == "naive":
                    naive_bkz_tour(g6k, tracer, blocksize,
                                   extra_dim4free=extra_dim4free,
                                   dim4free_fun=dim4free_fun,
                                   workout_params=workout_params,
                                   pump_params=pump_params)

                elif algbkz == "pump_and_jump":
                    pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                         dim4free_fun=dim4free_fun,
                                         extra_dim4free=extra_dim4free,
                                         pump_params=pump_params)
                elif algbkz == "slide":
                    slide_tour(g6k, dummy_tracer, blocksize, overlap=overlap,
                               dim4free_fun=dim4free_fun,
                               extra_dim4free=extra_dim4free,
                               workout_params=workout_params,
                               pump_params=pump_params)
                else:
                    raise ValueError("bkz/alg=%s not recognized." % algbkz)

            if verbose:
                slope = basis_quality(M)["/"]
                fmt = "{'alg': '%25s', 'jump':%2d, 'pds':%d, 'extra_d4f': %2d, 'beta': %2d, 'slope': %.5f, 'total walltime': %.3f}" # noqa
                print(fmt % (algbkz + "+" + ("enum" if algbkz == "fpylll" else g6k.params.default_sieve),
                             jump, pump_params["down_sieve"], extra_dim4free,
                             blocksize, slope, time.time() - T0))

    tracer.exit()
    slope = basis_quality(M)["/"]
    stat = tracer.trace
    try:
        stat.data["slope"] = np.array(slope)
        return stat
    except AttributeError:
        return None


def bkz_tour():
    """
    Run bkz tours.

    ..  note :: that by default no information is printed.
        To enable set ``--dummy-tracer False`` and ``--verbose``.

    """
    description = bkz_tour.__doc__

    args, all_params = parse_args(description,
                                  bkz__alg="pump_and_jump",
                                  bkz__blocksizes="40:51:2",
                                  bkz__pre_blocksize=39,
                                  bkz__tours=1,
                                  bkz__extra_dim4free=0,
                                  bkz__jump=1,
                                  bkz__dim4free_fun="default_dim4free_fun",
                                  slide__overlap=1,
                                  pump__down_sieve=True,
                                  challenge_seed=0,
                                  dummy_tracer=False,  # set to control memory
                                  verbose=False
                                  )

    stats = run_all(bkz_kernel, list(all_params.values()),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


    inverse_all_params = OrderedDict([(v, k) for (k, v) in six.iteritems(all_params)])
    stats = sanitize_params_names(stats, inverse_all_params)

    fmt = "{name:50s} :: n: {n:2d}, cputime {cputime:7.4f}s, walltime: {walltime:7.4f}s, slope: {slope:1.5f}, |db|: 2^{avg_max:.2f}"
    profiles = print_stats(fmt, stats, ("cputime", "walltime", "slope", "avg_max"),
                           extractf={"avg_max": lambda n, params, stat: db_stats(stat)[0]})

    output_profiles(args.profile, profiles)

    if args.pickle:
        pickler.dump(stats, open("bkz-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))


if __name__ == '__main__':
    bkz_tour()
