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
BKZ Tours.
"""
from __future__ import absolute_import
from __future__ import print_function
import sys
from .pump import pump
from .workout import workout
import six
from six.moves import range


def dim4free_wrapper(dim4free_fun, blocksize):
    """
    Deals with correct dim4free choices for edge cases when non default
    function is chosen.

    :param dim4free_fun: the function for choosing the amount of dim4free
    :param blocksize: the BKZ blocksize

    """
    if blocksize < 40:
        return 0
    dim4free = dim4free_fun(blocksize)
    return int(min((blocksize - 40)/2, dim4free))


def default_dim4free_fun(blocksize):
    """
    Return expected number of dimensions for free, from exact-SVP experiments.

    :param blocksize: the BKZ blocksize

    """
    return int(11.5 + 0.075*blocksize)


def naive_bkz_tour(g6k, tracer, blocksize, dim4free_fun=default_dim4free_fun,
                   extra_dim4free=0, workout_params=None, pump_params=None):
    """
    Run a naive BKZ-tour: call ``workout`` as an SVP oracle consecutively on
    each block.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param blocksize: dimension of the blocks
    :param dim4free_fun: number of dimension for free as a function of beta (function, or string e.g. `lambda x: 11.5+0.075*x`)
    :param extra_dim4free: increase the number of dims 4 free (blocksize is increased, but not sieve dimension)
    :param workout_params: parameters to pass to the workout
    :param pump_params: parameters to pass to the pump

    """
    if workout_params is None:
        workout_params = {}

    if "dim4free_min" in workout_params:
        raise ValueError("In naive_bkz, you should choose dim4free via dim4free_fun.")

    d = g6k.full_n

    if isinstance(dim4free_fun, six.string_types):
        dim4free_fun = eval(dim4free_fun)

    dim4free = dim4free_wrapper(dim4free_fun, blocksize) + extra_dim4free
    blocksize += extra_dim4free

    for kappa in range(d-3):
        beta = min(blocksize, d - kappa)
        lost_dim = blocksize - beta
        f = max(dim4free - lost_dim, 0)

        workout(g6k, tracer, kappa, beta, f, pump_params=pump_params, **workout_params)
        g6k.lll(0, d)


def pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=1,
                         dim4free_fun=default_dim4free_fun, extra_dim4free=0,
                         pump_params=None, goal_r0=0., verbose=False):
    """
    Run a PumpNjump BKZ-tour: call Pump consecutively on every (jth) block.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param blocksize: dimension of the blocks
    :param jump: only call the pump every j blocks
    :param dim4free_fun: number of dimension for free as a function of beta (function, or string
        e.g. `lambda x: 11.5+0.075*x`)
    :param extra_dim4free: increase the number of dims 4 free (blocksize is increased, but not sieve
        dimension)
    :param pump_params: parameters to pass to the pump
    """
    if pump_params is None:
        pump_params = {"down_sieve": False}

    if "dim4free" in pump_params:
        raise ValueError("In pump_n_jump_bkz, you should choose dim4free via dim4free_fun.")

    d = g6k.full_n
    g6k.shrink_db(0)
    g6k.lll(0, d)
    g6k.update_gso(0, d)


    if isinstance(dim4free_fun, six.string_types):
        dim4free_fun = eval(dim4free_fun)

    dim4free = dim4free_wrapper(dim4free_fun, blocksize) + extra_dim4free
    blocksize += extra_dim4free

    indices  = [(0, blocksize - dim4free + i, i) for i in range(0, dim4free, jump)]
    indices += [(i, blocksize, dim4free) for i in range(0, d - blocksize, jump)]
    indices += [(d - blocksize + i, blocksize - i, dim4free - i) for i in range(0, dim4free, jump)]

    pump_params["down_stop"] = dim4free + 3

    for (kappa, beta, f) in indices:
        if verbose:
            print("\r k:%d, b:%d, f:%d " % (kappa, beta, f), end=' ')
            sys.stdout.flush()

        pump(g6k, tracer, kappa, beta, f, **pump_params)
        g6k.lll(0, d)
        if g6k.M.get_r(0, 0) <= goal_r0:
            return

    if verbose:
        print("\r k:%d, b:%d, f:%d " % (d-(blocksize-dim4free), blocksize-dim4free, 0), end=' ')
        sys.stdout.flush()
    pump_params["down_stop"] = blocksize-dim4free
    pump(g6k, tracer, d-(blocksize-dim4free), blocksize-dim4free, 0, **pump_params)
    if verbose:
        print()

def slide_tour(g6k, tracer, blocksize, dim4free_fun=default_dim4free_fun, overlap = 1,
                   extra_dim4free=0, workout_params=None, pump_params=None):
    """
    Run a slide reduction tour: call ``workout`` as an SVP oracle consecutively on
    the disjoint blocks and then on the dual blocks shifted by ``overlap''

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param blocksize: dimension of the blocks
    :param dim4free_fun: number of dimension for free as a function of beta (function, or string e.g. `lambda x: 11.5+0.075*x`)
    :param overlap: shift of the dual blocks
    :param extra_dim4free: increase the number of dims 4 free (blocksize is increased, but not sieve dimension)
    :param workout_params: parameters to pass to the workout
    :param pump_params: parameters to pass to the pump

    """
    
    if workout_params is None:
        workout_params = {}

    if "dim4free_min" in workout_params:
        raise ValueError("In slide reduction, you should choose dim4free via dim4free_fun.")
    
    d = g6k.full_n
    # assuming blocksize | d for now
    if d - (d/blocksize)*blocksize > 0:
        raise ValueError("In slide reduction, the block size must divide the dimension.")
    
    if overlap >= blocksize:
        raise ValueError("In slide reduction, the overlap must be smaller than the blocksize.")

    if isinstance(dim4free_fun, six.string_types):
        dim4free_fun = eval(dim4free_fun)
    
    dim4free = dim4free_wrapper(dim4free_fun, blocksize) + extra_dim4free
    blocksize += extra_dim4free
    
    dm = g6k.params.dual_mode
    
    primal_blocks = range(0, d, blocksize)
    # compute and invert indices for dual blocks (since g6k inverts them)
    dual_blocks = [d - (x + overlap) for x in primal_blocks][1:]
    
    for forward in [True, False]:
        blocks = primal_blocks if forward else dual_blocks
        if not forward:
            dm = not dm
        with g6k.temp_params(dual_mode=dm):
            for kappa in blocks:
                beta = min(blocksize, d - kappa)
                lost_dim = blocksize - beta
                f = max(dim4free - lost_dim, 0)
                
                workout(g6k, tracer, kappa, beta, f, pump_params=pump_params, **workout_params)
                g6k.lll(0, d)
