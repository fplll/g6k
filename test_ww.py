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
    return M.to_canonical(tmp, start=M.d-offset)

n, betamax, sieve_dim = 96, 50, 50
B = IntegerMatrix(n,n)
B.randomize("qary", k=n//2, bits=11.705)
ft = "ld" if n<193 else "dd"
G = GSO.Mat(B, float_type=ft)
G.update_gso()

lll = LLL.Reduction(G)
lll()

bkz = LatticeReduction(B)
for beta in range(5,betamax+1):
    then_round=time.perf_counter()
    bkz.BKZ(beta,tours=5)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")

int_type = bkz.gso.B.int_type
G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
G.update_gso()
c = [ randrange(-3,4) for j in range(n) ]
e = np.array( [ randrange(-2,3) for j in range(n) ],dtype=np.int64 )
print(f"gauss: {gaussian_heuristic(G.r())**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ int(tt) for tt in t_ ]

print(f"ans: {b}")
print(f"ans_coords: {c}")
print(f"target: {t}")

# Gsave = gsomat_copy( G )

param_sieve = SieverParams()
param_sieve['threads'] = 5
# param_sieve['db_size_factor'] = 3.75
param_sieve['default_sieve'] = "bgj1"
g6k = Siever(G,param_sieve)
g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
g6k()
g6k.M.update_gso()

print(f"dbsize: {len(g6k)}")

# t_gs = g6k.M.from_canonical(t)
t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")

then = perf_counter()

debug_directives = 768 + 100 #768 + 105
out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000, debug_directives=debug_directives)

print(f"Slicer done in: {perf_counter()-then}")

if sieve_dim==n:
    out = to_canonical_scaled( G,out_gs,offset=sieve_dim )
    out = [round(tt) for tt in out]

    print(len(out))
    bab = G.B[-sieve_dim:].multiply_left( G.babai(t,n-sieve_dim,sieve_dim) )
    print(f"Babai outputs: {bab}")
    print(f"Slicer outputs: {[float(o) for o in out_gs]} | {len(out_gs)}")
    print(f"out: {out}")
    # print(f"out-b={out-b_}")
    # print(f"out-t={np.array(out)-np.array(t)}")

    print(f"error vec: {[int(ee) for ee in e]}")

    out = np.array(out)

    print(f"Error vector has been found: {all(out==e[-sieve_dim:])}")
else:
    out = to_canonical_scaled( G,out_gs,offset=sieve_dim )

    projerr = G.to_canonical( G.from_canonical(e,start=n-sieve_dim), start=n-sieve_dim)
    # print(f"projerrL {projerr}")
    # print(f"out: {out}")
    diff_v =  np.array(projerr)-np.array(out)
    # print(f"Diff btw. cvp and slicer: {diff_v}")

    # ans = G.B.multiply_left( G.babai( t-np.array(out) ) )
    N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
    N.update_gso()
    bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
    bab_0 = N.babai(tmp)

    bab_01=np.array( bab_0+bab_1 )
    print((f"recovered*B^(-1): {bab_0+bab_1}"))
    print(c)
    print(f"Coeffs of b found: {(c==bab_01)}")
    print(f"Succsess: {all(c==bab_01)}")

    # ans_0 = N.B[:sieve_dim].multiply_left( bab_0 )
    # ans_1 = G.B[n-sieve_dim:].multiply_left( bab_1 )
    # ans = np.array( ans_1 ) + np.array( ans_0 )
    #
    # print(f"ans_slicer: {ans}")
    # recovered_err = t-np.array(ans)
    # # print(f"out-b={out-b_}")
    # # print(f"out-t={np.array(out)-np.array(t)}")
    # print(f"recovered_err: {recovered_err}")
    #
    # print(f"Error vector has been found: {all(recovered_err==e)}")
