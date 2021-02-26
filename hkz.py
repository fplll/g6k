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
Full Sieve Command Line Client
"""

from __future__ import absolute_import
import pickle as pickler
from collections import OrderedDict

from g6k.algorithms.workout import workout, pump
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, db_stats
from g6k.utils.util import sanitize_params_names, print_stats, output_profiles
import six


def hkz_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)
    verbose = params.pop("verbose")

    pump_params = pop_prefixed_params("pump", params)
    workout_params = pop_prefixed_params("workout", params)

    verbose = params.pop("verbose")
    if verbose:
        workout_params["verbose"] = True
    challenge_seed = params.pop("challenge_seed")

    A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)

    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("hkz", n), start_clocks=True)

    # runs a workout woth pump-down down until the end
    workout(g6k, tracer, 0, n, pump_params=pump_params, verbose=verbose, **workout_params)
    #Just making sure
    pump(g6k, tracer, 15, n-15, 0, **pump_params)
    g6k.lll(0, n)

    return tracer.exit()


def hkz():
    """
    Attempt HKZ reduction. 
    """
    description = hkz.__doc__

    args, all_params = parse_args(description,
                                  challenge_seed=0,
                                  pump__down_sieve=True,
                                  pump__down_stop=9999,
                                  saturation_ratio=.8,
                                  pump__prefer_left_insert=10,
                                  workout__dim4free_min=0, 
                                  workout__dim4free_dec=15
                                  )

    stats = run_all(hkz_kernel, list(all_params.values()),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed
                    )

    inverse_all_params = OrderedDict([(v, k) for (k, v) in six.iteritems(all_params)])
    stats = sanitize_params_names(stats, inverse_all_params)

    fmt = "{name:50s} :: n: {n:2d}, cputime {cputime:7.4f}s, walltime: {walltime:7.4f}s, |db|: 2^{avg_max:.2f}"
    profiles = print_stats(fmt, stats, ("cputime", "walltime", "avg_max"),
                           extractf={"avg_max": lambda n, params, stat: db_stats(stat)[0]})

    output_profiles(args.profile, profiles)

    if args.pickle:
        pickler.dump(stats, open("hkz-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))


if __name__ == '__main__':
    hkz()
