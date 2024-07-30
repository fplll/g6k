"""
This file requires LatticeReduction.py and generateLWEInstance.py files in the root of repo.
"""
# from __future__ import absolute_import

import sys
import numpy as np
from lwe_gen import generateLWEInstance
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

import warnings
warnings.filterwarnings("ignore", message="Dimension of lattice is larger than maximum supported")

def from_canonical_scaled(M, t, offset):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
    """
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=np.float64 )
    r_ = np.array( [sqrt(tt) for tt in M.r()[-offset:]], dtype=np.float64 )
    return t_*r_

def iterative_slicer(db,M,t_0,sieve_dim):
    M = M
    n,d = M.B.nrows, M.B.ncols
    # t_ = M.from_canonical(t)
    # t_1 = t_[:d-sieve_dim]
    # t_2 = t_[d-sieve_dim:]

    # db = g6k.itervalues() #get coeffs of short vectors

    L = []
    # t_cur = [ np.array( sieve_dim*[0], dtype=np.int64 ), from_canonical_scaled(M,t,sieve_dim) ]
    t_cur = deepcopy(t_0)
    t_cur_nrm = t_cur[1]@t_cur[1]
    min_nrm_diff_sq = 2**65
    indmin = None
    ind = 0
    change_made = False
    for b in db: #create list of (coeffs, gso coeffs)
        tmp = from_canonical_scaled(M,M.B[-sieve_dim:].multiply_left(b[-sieve_dim:]),sieve_dim)
        L.append( [np.array( b[-sieve_dim:], dtype=np.int64 ), np.array( tmp, dtype=np.float64 )] )
        """
        For each L[-1] find min || t_cur - L[-1] ||
        """
    #     tmp = t_cur[1] - L[-1][1]
    #     tmp_nrm_sq = tmp@tmp
    #     if tmp_nrm_sq < min_nrm_diff_sq:
    #         min_nrm_diff_sq = tmp_nrm_sq
    #         ind_min = ind
    #         tmp_min = tmp
    #         change_made = True
    #     ind += 1
    # if change_made:
    #     t_cur = t_cur[0] - L[ind_min][0], tmp_min #substract closest to t vector from L
    #     change_made = False
    # print(f"Init. nrm. diff: {print(min_nrm_diff_sq**0.5)}")

    indmin = None
    ind = 0
    for reductions in range(2**10):
        for ind in range(len(L)): #create list of (coeffs, gso coeffs)
            # b = L[ind]
            """
            For each L[-1] find min || t_cur - L[-1] ||
            """
            if ind % 12000 == 0:
                tmp = M.B[-sieve_dim:].multiply_left( t_cur[0] )
                tmp = from_canonical_scaled(M,tmp,sieve_dim)
                t_cur = t_cur[0], tmp

            tmp = t_cur[1] - L[ind][1]
            tmp_nrm_sq = tmp@tmp
            if tmp_nrm_sq < min_nrm_diff_sq:
                sign = -1
                min_nrm_diff_sq = tmp_nrm_sq
                ind_min = ind
                tmp_min = tmp
                change_made = True

            """
            For each L[-1] find min || t_cur + L[-1] ||
            This should effectively double the size of L, but leads to numerical errors?
            """
            tmp = t_cur[1] + L[ind][1]
            tmp_nrm_sq = tmp@tmp
            if tmp_nrm_sq < min_nrm_diff_sq:
                sign = 1
                min_nrm_diff_sq = tmp_nrm_sq
                ind_min = ind
                tmp_min = tmp
                change_made = True

            ind += 1
        if change_made: #if vector became closer, apply the most suitable transform
            # if (reductions%20)==0:
                # print(f"Cur. nrm. diff: {min_nrm_diff_sq**0.5}")
            t_cur = t_cur[0] + sign*L[ind_min][0], tmp_min #substract closest to t vector from L
            change_made = False
        else: #else, we're done
            print(f"breaking!!!!!!!!!!!!!!!!!!!!!!!!!!!! {reductions}")
            break
    # print( t_cur )
    print(f"min nrm diff: {min_nrm_diff_sq**0.5}")
    return t_0[0] - t_cur[0], t_0[1] - t_cur[1]

def try_batch_cvp_w_babai( n, betamax=32, sieve_dim=32 ):
    """
    Generate LWE
    """
    A,b,q,s,e = generateLWEInstance(n)
    w = 1/q * (b - s.dot(A) - e)
    w = w.astype(int)

    n,m = A.shape
    d = n+m

    B = np.identity(d).astype(int)
    for i in range(m):
        B[i,i] *= int(q)

    for i in range(n):
        for j in range(m):
            B[m+i,j] =  int(A[i,j])

    x = np.concatenate( [w,s] )
    v = x.dot(B)
    e = np.concatenate( [e,-s] )
    t = x.dot(B) + e

    assert ( t == np.concatenate( [b,[0]*n] ) ).all()
    """
    Reduce the basis.
    """
    bkz = LatticeReduction(B)
    for beta in range(2,betamax+1):
        then_round=time.perf_counter()
        bkz.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")

    M = bkz.gso
    _ = M.update_gso()

    t_ = M.from_canonical(t)
    t_1 = t_[:d-sieve_dim]
    t_2 = t_[d-sieve_dim:]

    """
    Get a list of short vectors of the LWE lattice.
    """
    A = M.B
    param_sieve = SieverParams()
    param_sieve['threads'] = 10
    # param_sieve['db_size_base'] = 1.52
    # param_sieve['db_size_factor'] = 3.75
    g6k = Siever(M,param_sieve)
    llft, lft, rft = n-sieve_dim, n-min(45,sieve_dim), n
    g6k.initialize_local(llft, lft, rft)
    g6k()
    print( f"Init g6k db size: {len(g6k)}" )

    then = perf_counter()
    for shl in range(lft-llft): #shift the left bound of contxt to the left
        if shl%5==0:
            print( f"g6k db size: {len(g6k)} | curtime: {perf_counter()-then}" )
            print(f"Left context: {g6k.l}; right context: {g6k.r}")
        g6k.extend_left()
        g6k()
    g6k.update_gso(llft,rft)
    print( f"g6k db size: {len(g6k)} | curtime: {perf_counter()-then}" )
    # g6k.resize_db( ceil(2.44*len(g6k)), large=round(0.05*len(g6k)) )
    # g6k()
    # print( f"g6k db resize: {len(g6k)} | curtime: {perf_counter()-then}" )
    print(f"Left context: {g6k.l}; right context: {g6k.r}")

    print(f"e: {e}")
    print(f"s: {s}")
    print(f"x: {x}")
    print(f"error nrm: {(e@e)**0.5}")
    Uinv = M.UinvT.transpose()
    U = M.U
    tmp_ = Uinv.multiply_left(x)[-sieve_dim:]
    tmp_ = np.array( tmp_,dtype=np.int64 )
    print( [int(ttmp) for ttmp in tmp_] ) #last coords of answer?

    """
    Apply Iterative Slicer
    """
    for tries in range(25):
        """
        t_0 consists of 2 sieve_dim vectors:
          t_0[0] - vector storing transformation induced by the slicer. The coordinates are w.r.t.
          the basis B[-sieve_dim:]
          t_0[1] - coordinates to t_0 w.r.t. ( Q*diag(||b_i^*||^{-1/2} for i in range(d)) )[-sieve_dim:] -- orthonormal
          GS basis
        """
        t_0   = [ np.array( sieve_dim*[0], dtype=np.int64 ), from_canonical_scaled(M,t,sieve_dim) ]

        if tries!=0:
            """
            If not the first attempt, rerandomize targer with a vector b \in Lat(B[-sieve_dim:])
            """
            rnd = 3 #randrange(1,4)
            extra_err_c = [ [-1,1][randrange(2)] for ii in range(rnd) ] + (sieve_dim-rnd)*[0]
            shuffle(extra_err_c)
            extra_err_b = np.array( M.B.multiply_left( extra_err_c ),dtype=np.int64 ) #randomize target
            extra_err_b = from_canonical_scaled(M,extra_err_b,sieve_dim)
        else:
            extra_err_c = np.array( sieve_dim*[0],dtype=np.int64 )
            extra_err_b = np.array( sieve_dim*[0],dtype=np.float64 )

        """
        t2_ -- coordinates of b_ s.t. b_=t2_*B[-sieve_dim:] and proj_{d-sieve_dim}(t-b_) is small
        t2_gso -- || proj_{d-sieve_dim}(b_) ||
        """
        t2_, t2_gso = iterative_slicer(g6k.itervalues(),g6k.M,[t_0[0]-extra_err_c,t_0[1]-extra_err_b],sieve_dim)
        t2_, t2_gso = t2_+extra_err_c, t2_gso+extra_err_b

        #Recover x2 inductively via the identity x2_{beta-i} + \sum_{j=1}^i  R_{d-j,d-i} = v2_{beta-i}.
        v2_ = t2_gso * np.array( [1/sqrt(rr) for rr in M.r()[-sieve_dim:]] )
        x2 = [0]*sieve_dim
        for i in range(1,sieve_dim+1):
            x2[sieve_dim-i] = v2_[sieve_dim-i] - 1/ M.get_r(d-i, d-i) * sum( [ x2[sieve_dim-j] * M.get_r(d-j, d-i) for j in range(1,i) ] )
            x2[sieve_dim-i] = int( round( x2[sieve_dim-i] ) )
        # x2 = tuple(x2)
        print([int(xx) for xx in x2])
        succ = all( x2[i]==tmp_[i] for i in range(len(tmp_)) )
        if succ:
            print(f"Success!")
            break
        else:
            print(f"NO: {x2-tmp_}")


try_batch_cvp_w_babai( 50, betamax=40, sieve_dim=43 )
