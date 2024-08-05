"""
This file requires LatticeReduction.py and generateLWEInstance.py files in the root of repo.
"""
# from __future__ import absolute_import

import sys
import numpy as np
# from lwe_gen import generateLWEInstance
from LatticeReduction import LatticeReduction
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from math import sqrt, ceil, floor
from copy import deepcopy
from random import shuffle, randrange

import fpylll
print(f"fpylll version: {fpylll.__version__}")

import warnings
warnings.filterwarnings("ignore", message="Dimension of lattice is larger than maximum supported")

def from_canonical_scaled(M, t, offset=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if offset is None:
        offset=M.d
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=np.float128 )
    r_ = np.array( [sqrt(tt) for tt in M.r()[-offset:]], dtype=np.float128 )
    return t_*r_

def to_canonical_scaled(M, t, offset=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if offset is None:
        offset=M.d
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=np.float128 )
    r_ = np.array( [sqrt(tt)**(-0.5) for tt in M.r()[-offset:]], dtype=np.float128 )
    tmp = t_*r_
    return M.to_canonical(tmp)

n = 42
B = IntegerMatrix(n,n)
B.randomize("qary", k=n//2, bits=17.2)
G = GSO.Mat(B)
G.update_gso()

lll = LLL.Reduction(G)
lll()

bkz = LatticeReduction(B)
for beta in range(5,36):
    then_round=time.perf_counter()
    bkz.BKZ(beta,tours=5)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")

int_type = bkz.gso.B.int_type
G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type) )
G.update_gso()
c = [ randrange(-3,4) for j in range(n) ]
e = np.array( [ randrange(0,2) for j in range(n) ],dtype=np.int64 )

b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ int(tt) for tt in t_ ]

print(f"ans: {b}")
print(f"ans_coords: {c}")
print(f"target: {t}")

param_sieve = SieverParams()
#param_sieve['threads'] = 10
param_sieve['default_sieve'] = "bgj1"
g6k = Siever(G,param_sieve)
g6k.initialize_local(0,0,n)
g6k()

print(f"dbsize: {len(g6k)}")

# t_gs = from_canonical_scaled( g6k.M,t )
t_gs = g6k.M.from_canonical(t)

then = perf_counter()
out_gs = g6k.randomized_iterative_slice(t_gs,200)
print(f"Slicer done in: {perf_counter()-then}")

# out = to_canonical_scaled(g6k.M,out_gs)
out = g6k.M.to_canonical(out_gs)
out = [round(tt) for tt in out]
print(len(out))
print(f"Slicer outputs: {out}")
print(f"out-b={out-b_}")
print(f"out-t={np.array(out)-np.array(t)}")
