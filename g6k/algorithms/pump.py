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
import sys
import time
from math import log
from g6k.siever import SaturationError
import logging


def print_pump_state(pump):
    pump.minl = min(pump.g6k.l, pump.minl)
    if pump.phase != "down":
        print("\r %3d: ↑%3d      " % (pump.r-pump.l, pump.g6k.r-pump.g6k.l), end=' ')
    else:
        print("\r %3d: ↑%3d ↓%3d " % (pump.r-pump.l, pump.r-pump.minl, pump.r-pump.g6k.l), end=' ')
    sys.stdout.flush()


# Sieve (switching to gauss if needed) and test loop-breaking conditions
# Return true if we should continue
def wrapped_sieve(pump):
    if pump.phase == "init":
        alg = "gauss"
    else:
        alg = None

    cont = True
    try:
        with pump.g6k.temp_params(saturation_ratio=pump.g6k.params.saturation_ratio * pump.sat_factor):

            # Match lifting effort to insertion strategy
            pump.g6k(alg=alg, tracer=pump.tracer)

    except SaturationError as e:
        if pump.saturation_error == "skip":
            pump.down_sieve = False
            logging.info("saturation issue: breaking pump.")
            cont = False
        elif pump.saturation_error == "weaken":
            logging.info("saturation issue: weakening pump.")
            pump.sat_factor /= 2.
        elif pump.saturation_error == "ignore":
            pass
        else:
            raise e

    if pump.phase == "up" and (pump.max_up_time is not None):
        if pump.max_up_time < time.time() - pump.up_time_start:
            cont = False

    if pump.goal_r0 is not None:
        pump.g6k.insert_best_lift(scoring_goal_r0, aux=pump)

        if (pump.g6k.M.get_r(pump.kappa, pump.kappa) <= pump.goal_r0):
            cont = False

    return cont


def scoring_goal_r0(i, nlen, olen, aux):
    return i == aux.kappa and nlen < aux.goal_r0


def scoring_down(i, nlen, olen, aux):
    if i < aux.insert_left_bound or nlen >= olen:
        return False
    return log(olen / nlen) - i * log(aux.prefer_left_insert)


def pump(g6k, tracer, kappa, blocksize, dim4free, down_sieve=False,                                 # Main parameters
         goal_r0=None, max_up_time=None, down_stop=None, start_up_n=30, saturation_error="weaken",  # Flow control of the pump
         increasing_insert_index=True, prefer_left_insert=1.04,                                     # Insertion policy
         verbose=False,                                                                             # Misc
         ):
    """
    Run the pump algorithm.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param kappa: beginning of the block
    :param blocksize: dimension of the block (r=kappa+blocksize)
    :param dim4free: number of ``dimension for free'' [Ducas, Eurcrypt 2018]: Sieve context [l,r] where l=kappa+dim4free
    :param down_sieve: re-sieve after each insert during the pump-down phase.  (stronger reduction,
        slower running time)
    :param goal_r0: an extra hook to always insert at position kappa if this goal length can be met
        by a lift.  Quit when this is reached.
    :param max_up_time: For balancing BKZ time with SVP call time in LWE.  Stop pumping up when this
        time has elapsed.
    :param down_stop: stop inserts during pumping down after index kappa+down_stop (to control
        overheads of insert in large dimensional lattices)
    :param start_up_n: Initial sieve-context dimension for pumping up (starting at 1 incurs useless overheads)
    :param saturation_error: determines the behavior of pump when encountering saturation issue {"weaken",
        "skip", "ignore", "forward"}
    :param increasing_insert_index: During pump-down, always insert on the right side of the previous insertion.
    :param prefer_left_insert: Parameter theta from the paper (Sec 4.4) for scoring insertion candidates.
    :param verbose: print pump steps on the standard output.

    """
    pump.r = kappa+blocksize
    pump.l = kappa+dim4free  # noqa

    g6k.shrink_db(0)
    g6k.lll(kappa, pump.r)
    g6k.initialize_local(kappa, max(pump.r-start_up_n, pump.l+1), pump.r)

    pump.sat_factor = 1.
    pump.up_time_start = time.time()
    pump.insert_left_bound = kappa
    pump.minl = g6k.l

    for key in ('kappa', 'down_sieve', 'goal_r0', 'g6k', 'tracer',
                'max_up_time', 'saturation_error', 'verbose', 'prefer_left_insert'):
        setattr(pump, key, locals()[key])

    if down_stop is None:
        down_stop = dim4free

    with tracer.context(("pump", "beta:%d f:%d" % (blocksize, dim4free))):
        with g6k.temp_params(reserved_n=pump.r-pump.l):
            pump.phase = "init"
            wrapped_sieve(pump)  # The first initializing Sieve should always be Gauss to avoid rank-loss

            pump.phase = "up"
            # Pump Up
            while (g6k.l > pump.l):
                if(g6k.n + 1 > g6k.max_sieving_dim):
                    raise RuntimeError("The current sieving context is bigger than maximum supported dimension.")
                g6k.extend_left(1)

                if verbose:
                    print_pump_state(pump)
                if not wrapped_sieve(pump):
                    break

            if goal_r0 is not None and (g6k.M.get_r(kappa, kappa) <= goal_r0):
                return

            # Pump Down
            pump.phase = "down"
            while (g6k.n > 1) and (pump.insert_left_bound <= kappa+down_stop):
                # (try to) Insert
                ii = g6k.insert_best_lift(scoring_down, aux=pump)
                if ii is not None and increasing_insert_index:
                    pump.insert_left_bound = ii + 1

                else:
                    g6k.shrink_left(1)

                if goal_r0 is not None and (g6k.M.get_r(kappa, kappa) <= goal_r0):
                    break

                # Sieve (or Shrink db)
                if verbose:
                    print_pump_state(pump)
                if not pump.down_sieve:
                    g6k.resize_db(max(500, g6k.db_size() / g6k.params.db_size_base))
                elif not wrapped_sieve(pump):
                    break
