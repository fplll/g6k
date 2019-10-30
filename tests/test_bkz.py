# -*- coding: utf-8 -*-

from __future__ import absolute_import
from fpylll import IntegerMatrix
from g6k.algorithms.bkz import pump_n_jump_bkz_tour as bkz
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
