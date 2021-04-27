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



from __future__ import absolute_import
from __future__ import print_function
import copy

from fpylll.util import gaussian_heuristic

from g6k.algorithms.workout import workout
from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer
from g6k.utils.util import load_svpchallenge_and_randomize, save_svpchallenge_norm
from g6k.utils.util import sanitize_params_names, print_stats, output_profiles
from six.moves import range


def svp_kernel_trial(arg0, params=None, seed=None, goal_r0=None):
    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)
    dim4free_dec = params.pop("workout/dim4free_dec")
    pump_params = pop_prefixed_params("pump", params)
    challenge_seed = params.pop("challenge_seed")

    A, _ = load_svpchallenge_and_randomize(n, s=challenge_seed, seed=seed)
    g6k = Siever(A, params, seed=seed)
    tracer = SieveTreeTracer(g6k, root_label=("svp-challenge", n), start_clocks=True)

    gh = gaussian_heuristic([g6k.M.get_r(i, i) for i in range(n)])
    ds = list(range(0, n - 40, dim4free_dec))[::-1] + 10*[0]

    if goal_r0 is None:
        goal_r0 = 1.1 * gh

    for d in ds:
        workout(g6k, tracer, 0, n, dim4free_dec=dim4free_dec, goal_r0=goal_r0*1.001, pump_params=pump_params)

    tracer.exit()
    return int(g6k.M.get_r(0, 0)), gh


def svp_kernel(arg0, params=None, seed=None):
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    challenge_seed = params["challenge_seed"]

    goal_r0 = None
    matches = 0
    trials = 0
    while matches < 5:
        trials += 1
        found_r0, gh = svp_kernel_trial(arg0, goal_r0=goal_r0)
        if found_r0 == goal_r0:
            matches += 1
        else:
            matches = 0
            goal_r0 = found_r0
        print("\t", (n, challenge_seed), "Trial %3d, found norm %10d = %.4f*gh, consec matches %d/5" % (
            trials, goal_r0, goal_r0/gh, matches))

    save_svpchallenge_norm(n, goal_r0, s=challenge_seed)


def svp():
    """
    Run a progressive until 1.05-approx-SVP on matrices with dimensions in
    ``range(lower_bound, upper_bound, step_size)``.
    """
    description = svp.__doc__

    args, all_params = parse_args(description,
                                  workout__dim4free_dec=2,
                                  challenge_seed=0)

    run_all(svp_kernel, list(all_params.values()),
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            step_size=args.step_size,
            trials=args.trials,
            workers=args.workers,
            seed=args.seed)


if __name__ == '__main__':
    svp()
