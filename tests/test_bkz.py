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
from fpylll import IntegerMatrix
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
from g6k.algorithms.bkz import slide_tour as slide
from g6k.siever import Siever
from g6k.utils.stats import dummy_tracer

dimensions = (40, 50, 60)


def make_integer_matrix(d, int_type="mpz"):
    A = IntegerMatrix(d, d, int_type=int_type)
    A.randomize("qary", k=d//2, bits=10)
    return A


def test_bkz():
    for d in dimensions:
        # Primal
        A = make_integer_matrix(d)
        g6k = Siever(A)
        bkz(g6k, dummy_tracer, 20)
        bkz(g6k, dummy_tracer, 20)
        bkz(g6k, dummy_tracer, 20)

        # Dual
        A = make_integer_matrix(d)
        g6k = Siever(A)
        with g6k.temp_params(dual_mode=True):
            bkz(g6k, dummy_tracer, 20)
            bkz(g6k, dummy_tracer, 20)
            bkz(g6k, dummy_tracer, 20)

        # Primal then Dual
        A = make_integer_matrix(d)
        g6k = Siever(A)
        bkz(g6k, dummy_tracer, 20)
        bkz(g6k, dummy_tracer, 20)
        bkz(g6k, dummy_tracer, 20)
        with g6k.temp_params(dual_mode=True):
            bkz(g6k, dummy_tracer, 20)
            bkz(g6k, dummy_tracer, 20)
            bkz(g6k, dummy_tracer, 20)
        
        # Slide
        A = make_integer_matrix(d)
        g6k = Siever(A)
        block_size = 25 if d == 50 else 20
        slide(g6k, dummy_tracer, block_size, overlap = 10)
        slide(g6k, dummy_tracer, block_size, overlap = 10)
        slide(g6k, dummy_tracer, block_size, overlap = 10)
