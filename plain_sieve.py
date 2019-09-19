#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plain Sieve Command Line Client
"""

import logging
import pickle as pickler
from collections import OrderedDict


from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, db_stats


def plain_sieve_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    alg = params.pop("alg")

    reserved_n = n
    params = params.new(reserved_n=reserved_n, otf_lift=False)

    A, _ = load_svpchallenge_and_randomize(n, s=0, seed=seed)
    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("plain-sieve", n), start_clocks=True)
    g6k.initialize_local(0, n)
    g6k(alg=alg, tracer=tracer)
    tracer.exit()
    return tracer.trace


def plain_sieve():
    """
    Run a a plain sieve in the full dimension directly.
    """
    description = plain_sieve.__doc__

    args, all_params = parse_args(description,)

    stats = run_all(plain_sieve_kernel, list(all_params.values()),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)

    inverse_all_params = OrderedDict([(v, k) for (k, v) in all_params.items()])

    for (n, params) in stats:
        stat = stats[(n, params)]
        cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
        walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
        avr_db, max_db = db_stats(stat)
        fmt = "%48s :: m: %1d, n: %2d, cputime :%7.4fs, walltime :%7.4fs, avr_max |db|: 2^%2.2f, max_max db |db|: 2^%2.2f"  # noqa
        logging.info(fmt %(inverse_all_params[params], params.threads, n, cputime, walltime, avr_db, max_db))

    if args.pickle:
        pickler.dump(stats, open("plain-sieve-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))


if __name__ == '__main__':
    plain_sieve()
