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
SVP Challenge Solver Command Line Client
"""

from __future__ import absolute_import
from __future__ import print_function
import copy
import logging
import pickle as pickler
from collections import OrderedDict

from fpylll.util import gaussian_heuristic

from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, load_matrix_file, db_stats
from g6k.utils.util import sanitize_params_names, print_stats, output_profiles

import six
from six.moves import range


def asvp_kernel(arg0, params=None, seed=None):
    logger = logging.getLogger("asvp")

    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    load_matrix = params.pop("load_matrix")
    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)
    verbose = params.pop("verbose")
    if verbose:
        workout_params["verbose"] = True
    challenge_seed = params.pop("challenge_seed")

    if load_matrix is None:
        A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)
        if verbose:
            print(("Loaded challenge dim %d" % n))
    else:
        A, _ = load_matrix_file(load_matrix)
        if verbose:
            print(("Loaded file '%s'" % load_matrix))

    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("svp-challenge", n), start_clocks=True)

    gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(n)])
    goal_r0 = (1.05 ** 2) * gh
    if verbose:
        print(
            (
                "gh = %f, goal_r0/gh = %f, r0/gh = %f"
                % (gh, goal_r0 / gh, sum([x * x for x in A[0]]) / gh)
            )
        )

    flast = workout(
        g6k, tracer, 0, n, goal_r0=goal_r0, pump_params=pump_params, **workout_params
    )

    tracer.exit()
    stat = tracer.trace
    stat.data["flast"] = flast

    if verbose:
        logger.info("sol %d, %s" % (n, A[0]))

    norm = sum([x * x for x in A[0]])
    if verbose:
        logger.info("norm %.1f ,hf %.5f" % (norm ** 0.5, (norm / gh) ** 0.5))

    return tracer.trace


def asvp():
    """
    Run a Workout until 1.05-approx-SVP on matrices with dimensions in ``range(lower_bound, upper_bound, step_size)``.
    """
    description = asvp.__doc__

    args, all_params = parse_args(
        description,
        load_matrix=None,
        verbose=True,
        challenge_seed=0,
        workout__dim4free_dec=3,
    )

    stats = run_all(
        asvp_kernel,
        list(all_params.values()),
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        step_size=args.step_size,
        trials=args.trials,
        workers=args.workers,
        seed=args.seed,
    )

    inverse_all_params = OrderedDict([(v, k) for (k, v) in six.iteritems(all_params)])
    stats = sanitize_params_names(stats, inverse_all_params)

    fmt = "{name:50s} :: n: {n:2d}, cputime {cputime:7.4f}s, walltime: {walltime:7.4f}s, flast: {flast:3.2f}, |db|: 2^{avg_max:.2f}"
    profiles = print_stats(
        fmt,
        stats,
        ("cputime", "walltime", "flast", "avg_max"),
        extractf={"avg_max": lambda n, params, stat: db_stats(stat)[0]},
    )

    output_profiles(args.profile, profiles)

    if args.pickle:
        pickler.dump(
            stats,
            open(
                "svp-challenge-%d-%d-%d-%d.sobj"
                % (args.lower_bound, args.upper_bound, args.step_size, args.trials),
                "wb",
            ),
        )


if __name__ == "__main__":
    asvp()
