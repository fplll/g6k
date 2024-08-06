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

def gsomat_copy(M):
    n,m,int_type,float_type = M.B.nrows,M.B.ncols,M.int_type,M.float_type

    B = []
    for i in range(n):
        B.append([])
        for j in range(m):
            B[-1].append(int(M.B[i][j]))
    B = IntegerMatrix.from_matrix( B,int_type=int_type )

    U = []
    for i in range(n):
        U.append([])
        for j in range(m):
            U[-1].append(int(M.U[i][j]))
    U = IntegerMatrix.from_matrix( U,int_type=int_type )

    UinvT = []
    for i in range(n):
        UinvT.append([])
        for j in range(m):
            UinvT[-1].append(int(M.UinvT[i][j]))
    UinvT = IntegerMatrix.from_matrix( UinvT,int_type=int_type )

    M = GSO.Mat( B, float_type=float_type, U=U, UinvT=UinvT )
    M.update_gso()
    return M

def from_canonical_scaled(M, t, offset=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if offset is None:
        offset=M.d
    gh = gaussian_heuristic(M.r()[-offset:])
    print(f"gh: {gh}")
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=np.float64 )
    r_ = np.array( [sqrt(tt/gh) for tt in M.r()[-offset:]], dtype=np.float64 )

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

    gh = gaussian_heuristic(M.r()[-offset:])**0.5
    r_ = np.array( [1/sqrt(tt) for tt in M.r()[-offset:]], dtype=np.float64 ) * gh
    tmp = t*r_
    return M.to_canonical(tmp)[-offset:]

n = 42
B = IntegerMatrix(n,n)
B.randomize("qary", k=n//2, bits=14.05)
G = GSO.Mat(B)
G.update_gso()

lll = LLL.Reduction(G)
lll()

bkz = LatticeReduction(B)
for beta in range(5,43):
    then_round=time.perf_counter()
    bkz.BKZ(beta,tours=5)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")

int_type = bkz.gso.B.int_type
G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type) )
G.update_gso()
c = [ randrange(-3,4) for j in range(n) ]
e = np.array( [ randrange(-2,3) for j in range(n) ],dtype=np.int64 )

b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ int(tt) for tt in t_ ]

print(f"ans: {b}")
print(f"ans_coords: {c}")
print(f"target: {t}")

# Gsave = gsomat_copy( G )

param_sieve = SieverParams()
#param_sieve['threads'] = 10
param_sieve['default_sieve'] = "bgj1"
g6k = Siever(G,param_sieve)
g6k.initialize_local(0,0,n)
g6k()
g6k.M.update_gso()

print(f"dbsize: {len(g6k)}")

# t_gs = g6k.M.from_canonical(t)
t_gs = from_canonical_scaled( G,t )
print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")

then = perf_counter()

out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=50)

print(f"Slicer done in: {perf_counter()-then}")

out = to_canonical_scaled( G,out_gs )
# out = g6k.M.to_canonical(out_gs)
# print(out)
out = [round(tt) for tt in out]

print(len(out))
bab = G.B.multiply_left( G.babai(t) )
print(f"Babai outputs: {bab}")
print(f"Slicer outputs: {[float(o) for o in out_gs]}")
print(f"out: {out}")
print(f"out-b={out-b_}")
print(f"out-t={np.array(out)-np.array(t)}")

print(f"error: {[int(ee) for ee in e]}")

out = np.array(out)

print(f"Error vector has been found: {all(out==e)}")
