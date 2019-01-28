#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import copy
import logging
import pickle as pickler
from collections import OrderedDict

from fpylll.util import gaussian_heuristic

from g6k.algorithms.ducas18 import ducas18

from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, load_svpchallenge_norm, db_stats


from fpylll import BKZ as fplll_bkz
from fpylll.tools.bkz_stats import dummy_tracer
from fpylll import Enumeration, EnumerationError


GRADIENT_BLOCKSIZE = 31
NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]


# Re-implement bkz2.svp_reduction, with a precise radius goal rather than success proba
def svp_enum(bkz, params, goal):
    n = bkz.M.d
    r = [bkz.M.get_r(i, i) for i in range(0, n)]
    gh = gaussian_heuristic(r)

    rerandomize = False
    while bkz.M.get_r(0, 0) > goal:
        if rerandomize:
            bkz.randomize_block(0, n)
        bkz.svp_preprocessing(0, n, params)

        strategy = params.strategies[n]
        radius = goal
        pruning = strategy.get_pruning(goal, gh)

        try:
            enum_obj = Enumeration(bkz.M)
            max_dist, solution = enum_obj.enumerate(0, n, radius, 0, pruning=pruning.coefficients)[0]
            bkz.svp_postprocessing(0, n, solution, tracer=dummy_tracer)
            rerandomize = False
        except EnumerationError:
            rerandomize = True

        bkz.lll_obj()

    return


def svp_kernel(arg0, params=None, seed=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)
    challenge_seed = params.pop("challenge_seed")
    alg = params.pop("svp/alg")
    workout_params = pop_prefixed_params("workout/", params)
    pump_params = pop_prefixed_params("pump/", params)

    goal_r0 = 1.001 * load_svpchallenge_norm(n, s=challenge_seed)
    A, bkz = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)
    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("svp-exact", n), start_clocks=True)

    if alg == "enum":
        assert len(workout_params) + len(pump_params) == 0
        bkz_params = fplll_bkz.Param(block_size=n, max_loops=1, strategies=fplll_bkz.DEFAULT_STRATEGY,
                                     flags=fplll_bkz.GH_BND)
        svp_enum(bkz, bkz_params, goal_r0)
        flast = -1
    elif alg == "duc18":
        assert len(workout_params) + len(pump_params) == 0
        flast = ducas18(g6k, tracer, goal=goal_r0)
    elif alg == "workout":
        flast = workout(g6k, tracer, 0, n, goal_r0=goal_r0, pump_params=pump_params, **workout_params)
    else:
        raise ValueError("Unrecognized algorithm for SVP")

    r0 = bkz.M.get_r(0, 0) if alg == "enum" else g6k.M.get_r(0, 0)
    if r0 > goal_r0:
        raise ValueError('Did not reach the goal')
    if 1.002 * r0 < goal_r0:
        raise ValueError('Found a vector shorter than the goal for n=%d s=%d.'%(n, challenge_seed))

    tracer.exit()
    stat = tracer.trace
    stat.data["flast"] = flast
    return stat


def svp():
    """
    Run a progressive until exact-SVP is solved.

    The exact-SVP length must have been priorly determined using ./svp_exact_find_norm.py

    """
    description = svp.__doc__

    args, all_params = parse_args(description,
                                  challenge_seed=0,
                                  svp__alg="workout"
                                  )

    stats = run_all(svp_kernel, all_params.values(),
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)

    inverse_all_params = OrderedDict([(v, k) for (k, v) in all_params.iteritems()])

    stats2 = OrderedDict()
    for (n, params), v in stats.iteritems():
        params_name = inverse_all_params[params]
        params_name = re.sub("'challenge_seed': [0-9]+,", "", params_name)
        params = params.new(challenge_seed=None)
        stats2[(n, params_name)] = stats2.get((n, params_name), []) + v
    stats = stats2

    for (n, params) in stats:
        stat = stats[(n, params)]
        cputime = sum([float(node["cputime"]) for node in stat])/len(stat)
        walltime = sum([float(node["walltime"]) for node in stat])/len(stat)
        flast = sum([float(node["flast"]) for node in stat])/len(stat)
        avr_db, max_db = db_stats(stat)
        fmt = "%100s :: n: %2d, cputime :%7.4fs, walltime :%7.4fs, flast : %2.2f, , avr_max db: 2^%2.2f, max_max db: 2^%2.2f" # noqa
        logging.info(fmt % (params, n, cputime, walltime, flast, avr_db, max_db))

    if args.pickle:
        pickler.dump(stats, open("hkz-svp-%d-%d-%d-%d.sobj" %
                                 (args.lower_bound, args.upper_bound, args.step_size, args.trials), "wb"))


if __name__ == '__main__':
    svp()
