"""
This file requires LatticeReduction.py file in the root of repo.
"""
# from __future__ import absolute_import

import sys, os
import glob #for automated search in subfolders
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

import pickle
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

save_folder = "./saved_lattices/"

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

def gen_and_pickle_lattice(n, k=None, bits=None, betamax=None, seed=None):
    isExist = os.path.exists(save_folder)
    if not isExist:
        try:
            os.makedirs(save_folder)
        except:
            pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.


    k = n//2+1 if k is None else k
    bits = 11.705 if bits is None else bits
    betamax = 40 if betamax is None else betamax
    seed = randrange(2**32) if seed is None else seed

    B = IntegerMatrix(n,n)
    B.randomize("qary", k=n//2, bits=11.705)
    ft = "ld" if n<193 else "dd"

    try:
        G = GSO.Mat(B, float_type=ft)
    except: #if "dd" is not available
        FPLLL.set_precision(208)
        G = GSO.Mat(B, float_type="mpfr")
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

    param_sieve = SieverParams()
    param_sieve['threads'] = 5
    # param_sieve['db_size_factor'] = 3.75
    param_sieve['default_sieve'] = "bgj1"
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(0,0,n)
    then = perf_counter()
    g6k()
    print(f"Sieving in dim-{n} done in {perf_counter()-then}")
    g6k.M.update_gso()

    filename = save_folder + f"siever_{n}_{betamax}_{hex(seed)[2:]}.pkl"
    g6k.dump_on_disk(filename)
    print("Dump succsess")
    print(f"len g6k: {len(g6k)}")

def load_lattices(n):
    # An iterator through n-dimensional lattices
    # Each instance requires an exponential amount of memory, so we
    # don't store it all simoultaniously.
    # lats = []
    for filename in glob.glob(f'{save_folder}siever_{n}*.pkl'):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f: # open in readonly mode
            g6k_obj = Siever.restore_from_file(filename)
            yield g6k_obj
            # lats.append(L)
        print(filename)

def run_exp(g6k_obj, approx_fact=0.5, n_targets=1):
    B, G, n = g6k_obj.M.B, g6k_obj.M, g6k_obj.M.d
    gh = gaussian_heuristic(g6k_obj.M.r())
    max_unif = floor( gh / (8*n) * approx_fact ) #||e|| ~ approx_fact * E(\lambda_1(B)/2)
    assert max_unif, f"Zero error!"

    exp_results = {(n,approx_fact): []}
    for bdds in range(n_targets):
        c = [ randrange(-3,4) for j in range(n) ] #coeffs of answer w.r.t. B
        e = np.array( [ randrange(-max_unif,max_unif+1) for j in range(n) ],dtype=np.int64 )
        print(f"gauss: {gaussian_heuristic(G.r())**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t_ = e+b_
        t = [ int(tt) for tt in t_ ]

        t_gs = from_canonical_scaled( G,t )
        projerr = from_canonical_scaled( G,e,offset=n )
        projerr_sq_nrm = projerr@projerr
        print(f"projerr_sq_nrm: {projerr_sq_nrm}")

        then = perf_counter()

        stats_accumulator = {}
        out_gs = g6k_obj.randomized_iterative_slice([float(tt) for tt in t_gs],samples=10**5, dist_sq_bnd = projerr_sq_nrm, stats_accumulator=stats_accumulator) #1.01*projerr_sq_nrm

        sli_time = perf_counter()-then
        print(f"Slicer done in: {sli_time}")
        n_rerand_sli = stats_accumulator['n_rerand_sli']

        # - - - Verify the answer - - -
        out = to_canonical_scaled( G,out_gs )
        out = [round(tt) for tt in out]

        print(f"error vec: {[int(ee) for ee in e]}")

        out = np.array(out)

        succ = all(out==e)
        print(f"Error vector has been found: {succ}")

        exp_results[(n,approx_fact)].append( [n_rerand_sli, succ, sli_time] )
    print( exp_results )
    return exp_results


if __name__ == "__main__":
    n_workers = 2
    n_lats = 5
    n_instances_per_lattice = 5
    n, k, bits, betamax = 70, 35, 13.8, 40

    # - - - Generate lattices - - -
    # pool = Pool(processes = n_workers )
    # tasks = []
    # for seed in range(n_lats):
    #     tasks.append( pool.apply_async(
    #         gen_and_pickle_lattice, (n, k, bits, betamax, seed)
    #     ) )
    #
    # for t in tasks:
    #     t.get()

    # - - - Run experiments - - -
    lats = load_lattices(n)

    l = []
    for g6k_obj in lats:
        l.append( run_exp(g6k_obj, approx_fact=0.5, n_targets=100) )

    for ll in l:
        print(ll)

    D = l[0]
    for ll in l[1:]:
        for key in ll.keys():
            if not key in D.keys():
                D[key] = []
            D[key] += ll[key]

    n_rerand_sli_list, sli_time_list = [], []
    for key in D.keys():
        stats = D[key]

        n_rerand_sli_list += [ tt[0] for tt in stats]
        sli_time_list += [ tt[2] for tt in stats]

        print(f"n_rerand_sli_list:")
        print(f"    avg: {np.average(n_rerand_sli_list)}, med: {np.median(n_rerand_sli_list)}, min: {np.min(n_rerand_sli_list)}, max: {np.max(n_rerand_sli_list)} std: {np.std(n_rerand_sli_list)}")
        print()
        print(f"sli_time_list:")
        print(f"    avg: {np.average(sli_time_list)}, med: {np.median(sli_time_list)}, std: {np.std(sli_time_list)}")
        print()