"""
This file requires LatticeReduction.py, lwe_gen.py files in the root of repo.
"""
# from __future__ import absolute_import

import sys, os
import numpy as np
from lwe_gen import generateLWEInstance, binomial_vec, uniform_vec
from LatticeReduction import LatticeReduction
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from math import sqrt, ceil, floor, log, exp
from copy import deepcopy
from random import shuffle, randrange

from discretegauss import sample_dgauss

import pickle
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from g6k.siever import SaturationError
from utils import *

def run_exp(g6k_obj, approx_fact=1.0, n_targets=1, debug_directives = 873):
    size_t_k = ( debug_directives & 0xFF00 ) >> 8
    XPC_SLICER_SAMPLING_THRESHOLD_OVERRIDE = debug_directives & 0xFF

    B, G, n = g6k_obj.M.B, g6k_obj.M, g6k_obj.M.d
    gh = gaussian_heuristic(g6k_obj.M.r())
    sigma2 = ( gh / (4*n) ) * approx_fact**2 #||e|| ~ approx_fact * E(\lambda_1(B)/2)
    print(f"approx_fact: {approx_fact}")
    # assert max_unif > 0, f"Zero error!"

    exp_results = {(n,approx_fact,size_t_k,XPC_SLICER_SAMPLING_THRESHOLD): []}
    for bdds in range(n_targets):
        c = [ randrange(-3,4) for j in range(n) ] #coeffs of answer w.r.t. B
        e = np.array( [ sample_dgauss(sigma2) for j in range(n) ], dtype=np.int64 )
        # some other possible distributions:
        # e = binomial_vec( n,eta )
        # e = np.array( [ randrange(-max_unif,max_unif+1) for j in range(n) ],dtype=np.int64 )
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
        out_gs = g6k_obj.randomized_iterative_slice([float(tt) for tt in t_gs],samples=10**5, dist_sq_bnd = projerr_sq_nrm, stats_accumulator=stats_accumulator, debug_directives=debug_directives) #1.01*projerr_sq_nrm

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

        exp_results[(n,approx_fact,size_t_k,XPC_SLICER_SAMPLING_THRESHOLD)].append( [n_rerand_sli, succ, sli_time] )
    print( exp_results )
    return exp_results

def batchCVPP_prob(d, alpha, gamma):
    # as per https://wesselvanwoerden.com/publication/ducas-2020-randomized/randomized_slicer.pdf
    #https://eprint.iacr.org/2016/888.pdf gives the same numbers since we have sqrt(4/3) short vectors given
    #alpha = sqrt(4/3.) #1.00995049383621 #<2*c - 2*sqrt(c^2-c) < 1.10468547559928 since a <1.22033 (see p.22)
    #gamma = 1     #gamma-CVP
    a = alpha**2
    b = a**2/(4*a - 4)
    c = gamma**2
    assert (b>c), f"Bad b: {b} <= {c}"
    n = ceil(-1/2 + sqrt((4*b-a)**2-8*c*(2*b-a))/(2*a))

    #print(a, b, c, n)

    #Eq.(12)
    def p(a, x, y):
        return sqrt(a - (a+x-y)**2/(4*x))

    #Eq.(16)
    def omega(a, x, y):
        return -log(p(a,x,y))

    if n==1:
        u = 0
        v = c-b

    else:
        disc = (a*n**2 - (b+c))**2 + 4*b*c*(n**2-1)
        u = ( (b+c-a)*n - sqrt(disc) )/(n**3 - n)
        v = ( (a-2*b)*n**2 + (b-c) + n*sqrt(disc) )/(n**3-n)


    # Eq.(43) success probablity for one target
    prob = 0
    x_ = [0]*(n+1)
    x_[0] = b
    x_[n] = c
    for i in range(1, n+1):
        x_[i] = u*i**2+v*i+b
        prob += omega(a, x_[i-1], x_[i])
        #print(i, x_[i].n(), prob)

    # T = (a - 2*(a-1)/(1+sqrt(1-1./a)))**(-1/2.)  #base for power-d, runtime per instance!

    prob = exp(-prob) #1/prob=number of rerandomizations per target, base for power-d
    return prob

if __name__ == "__main__":
    n_workers = 2
    n_lats = 2
    n_instances_per_lattice = 2
    n, bits, betamax = 50, 13.8, 35
    k = n//2

    # - - - Generate lattices - - -
    pool = Pool(processes = n_workers )
    tasks = []
    for seed in range(n_lats):
        tasks.append( pool.apply_async(
            gen_and_pickle_lattice, (n, k, bits, betamax, seed)
        ) )

    for t in tasks:
        t.get()
    pool.close()
    # - - - Run experiments - - -
    lats = load_lattices(n)
    l = []
    for g6k_obj in lats:
        for XPC_SLICER_SAMPLING_THRESHOLD in [105]: #100, 105, 110
            l.append( run_exp(g6k_obj, approx_fact=0.98, n_targets=n_instances_per_lattice, debug_directives = XPC_SLICER_SAMPLING_THRESHOLD+0x300) )

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
        print(key)

        n_rerand_sli_list += [ tt[0] for tt in stats]
        sli_time_list += [ tt[2] for tt in stats]

        print(f"n_rerand_sli_list:")
        print(f"    avg: {np.average(n_rerand_sli_list)}, med: {np.median(n_rerand_sli_list)}, min: {np.min(n_rerand_sli_list)}, max: {np.max(n_rerand_sli_list)}, std: {np.std(n_rerand_sli_list)}")
        print()
        print(f"sli_time_list:")
        print(f"    avg: {np.average(sli_time_list)}, med: {np.median(sli_time_list)}, max: {np.max(sli_time_list)}, std: {np.std(sli_time_list)}")
        print()
        print(f"vs predicted: {batchCVPP_prob(n, sqrt(4./3.), 1.)**(-n/2.)}")
        print()
