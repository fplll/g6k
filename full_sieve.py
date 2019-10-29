#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Sieve Command Line Client
"""

from __future__ import absolute_import
import logging
import pickle as pickler
from collections import OrderedDict

import re

from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, db_stats
import six
from math import log
import numpy as np
import sys


def full_sieve_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    pump_params = pop_prefixed_params("pump", params)
    verbose = params.pop("verbose")

    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)

    challenge_seed = params.pop("challenge_seed")
    A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)

    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("full-sieve", n), start_clocks=True)

    # Actually runs a workout with very large decrements, so that the basis is kind-of reduced
    # for the final full-sieve
    workout(g6k, tracer, 0, n, dim4free_min=0, dim4free_dec=15, pump_params=pump_params, verbose=verbose)
    g6k.update_gso(0, n)

    tracer.exit()
    stat = tracer.trace
    stat.data["profile"] = np.array([log(g6k.M.get_r(i, i)) for i in range(n)])

    return stat


def full_sieve():
    """
    Run a a full sieve (with some partial sieve as precomputation).
    """
    description = full_sieve.__doc__

    args, all_params = parse_args(description,
                                  challenge_seed=0)

    stats = run_all(full_sieve_kernel, list(all_params.values()),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed                    
                    )

    inverse_all_params = OrderedDict([(v, k) for (k, v) in six.iteritems(all_params)])

    stats2 = OrderedDict()
    for (n, params), v in six.iteritems(stats):
        params_name = inverse_all_params[params]
        params_name = re.sub("'challenge_seed': [0-9]+,", "", params_name)
        params = params.new(challenge_seed=None)
        stats2[(n, params_name)] = stats2.get((n, params_name), []) + v
    stats = stats2


    for (n, params) in stats:
        stat = stats[(n, params)]
        cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
        walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
        avr_profile = sum([node["profile"] for node in stat])/len(stat)

        avr_db, max_db = db_stats(stat)
        fmt = "%100s :: n: %2d, cputime :%7.4fs, walltime :%7.4fs, , avr_max db: 2^%2.2f, max_max db: 2^%2.2f" # noqa
        logging.info(fmt % (params, n, cputime, walltime, avr_db, max_db))

        if args.profile:
            import matplotlib.pyplot as plt
            L = [x for x in avr_profile]
            plt.plot(L, label=params)
            print(L)


    if args.pickle:
        pickler.dump(stats, open("full-sieve-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))
    if args.profile:
        plt.legend()
        plt.show()


if __name__ == '__main__':
    full_sieve()
