import fpylll
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL
from time import perf_counter
import numpy as np

import sys, os
import glob #for automated search in subfolders

from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from math import sqrt, ceil, floor, log, exp
from copy import deepcopy
from random import shuffle, randrange

import pickle
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from LatticeReduction import LatticeReduction
from utils import * #uniform_random_distribution, reduce_to_fund_par_proj

def gen_cvpp_g6k(n,betamax=None,k=None,bits=11.705):
    betamax=n if betamax is None else betamax
    k = n//2 if k is None else k
    B = IntegerMatrix(n,n)
    # B.randomize("qary", k=k, bits=bits)
    B.randomize("uniform", bits=bits)

    LR = LatticeReduction( B )
    for beta in range(5,betamax+1):
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}")

    G = LR.gso
    param_sieve = SieverParams()
    param_sieve['threads'] = 2
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(0,0,n)
    print("Running bdgl2...")
    then=perf_counter()
    g6k(alg="bdgl2")
    print(f"bdgl2-{n} done in {perf_counter()-then}")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")
    return g6k

def run_exp(g6k,ntests,approx_facts):
    G = g6k.M
    B = g6k.B
    n = G.d

    lambda1 = G.get_r(0, 0)**0.5
    D = {}
    Ds = []
    for approx_fact in approx_facts:
        succ_enum, succ_slic, succ_bab = 0, 0, 0
        for tstnum in range(ntests):
            print(f" - - - {approx_fact} #{tstnum} out of {ntests} - - -")
            c = [ randrange(-2,3) for j in range(n) ]
            e = np.array( uniform_random_distribution(n,approx_fact*lambda1/2) )
            b = np.array( B.multiply_left( c ) )
            t = b+e
            print(e@e, 0.25*G.get_r(0, 0))

            """
            Testing Babai.
            """
            then = perf_counter()
            ctmp = G.babai( t )
            tmp = B.multiply_left( ctmp )
            print(f"Babai-{n} done in {perf_counter()-then}")
            err = tmp-b
            if not ( (err@err)<10**-6 ):
                print(f"FAIL after babai: {(err@err)}")
            else:
                succ_bab += 1

            """
            Testing Enumeration.
            """
            if n<55: #we do not run enumeration in dimensions >45
                then = perf_counter()
                # tmp = fpylll.CVP.closest_vector(B,[float(tt) for tt in t],method="proved")
                enum = Enumeration(G, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, nr_solutions=1)
                try:
                    tmp = enum.enumerate( 0, n, 1.01*(approx_fact*lambda1/2)**2, 0, target=G.from_canonical(t) )
                    tmp = tmp[0][1]
                    v = B.multiply_left( tmp )
                    print( np.array(c)-np.array(tmp) )
                    print(f"CVP-{n} done in {perf_counter()-then}")
                    sys.stdout.flush()
                    err = v-b
                    if not ( (err@err)<10**-6 ):
                        print(f"FAIL after enumeration: {(err@err)}")
                    else:
                        succ_enum += 1
                except EnumerationError:
                    print("Enum failed...")
            else:
                print(f"n={n} is too large for enumeration. Skipping...")

            """
            Testing Slicer.
            """
            t_gs = from_canonical_scaled( G,t,offset=n )

            #get the basis of full-dim projective lattice. Here, it is equal to B-rotated, but we need
            #to simulate the effect of floating points, so we still do that.
            B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=n), dtype=np.float64 ) for i in range(n) ] #
            t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),n) #reduce the target w.r.t. B_gs
            t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

            try:
                slicer = RandomizedSlicer(g6k)
                slicer.set_nthreads(2);
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=2000)

                blocks = 2 # should be the same as in siever
                blocks = min(3, max(1, blocks))
                blocks = min(int(n / 28), blocks)
                sp = SieverParams()
                N = sp["db_size_factor"] * sp["db_size_base"] ** n
                buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
                buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
                buckets = max(buckets, 2**(blocks-1))

                print("blocks: ", blocks, " buckets: ", buckets )

                then = perf_counter()
                slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (1.01*(e@e)))
                iterator = slicer.itervalues_t()
                print(f"Slicer done in {perf_counter()-then}")

                for tmp in iterator:
                    out_gs_reduced = tmp  #cdb[0]
                    break
                out_gs = out_gs_reduced + t_gs_shift
                projerr = G.to_canonical( G.from_canonical(e,start=0), start=0)
                print(projerr-out_gs)

                # out = to_canonical_scaled( G,out_gs,offset=n ) #returns error vector, if succseeds
                # v = t-out #now v is (up to precision) in lattice
                # vc = G.babai(v)
                # v = B.multiply_left(vc)
                # print( np.array(c)-np.array(tmp) )
                # print(f"CVP-{n} done in {perf_counter()-then}")
                # sys.stdout.flush()
                # err = v-b

                # if not ( (err@err)<10**-6 ):
                #     print(f"FAIL after slicer: {(err@err)}")
                # else:
                #     succ_slic += 1

                out = to_canonical_scaled( G,out_gs,offset=n )
                projerr = G.to_canonical( G.from_canonical(e,start=0), start=0)

                # N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
                # N.update_gso()
                bab_1 = G.babai(t-np.array(out),start=0) #last sieve_dim coordinates of s

                bab_01=np.array( bab_1 )
                succ = all(c==bab_01)
                print(f"Slic Succsess: {succ}")
                if not ( succ ):
                    print(f"FAIL after slicer: {(err@err)}")
                else:
                    succ_slic += 1
            except Exception as excpt: #if slicer fails for some reason,
                #then prey, this is not a devastating segfault
                print(excpt)
                raise excpt
                # pass

        D[(n,approx_fact)] = (1.0*succ_enum / ntests, 1.0*succ_slic / ntests, 1.0*succ_bab / ntests)
        Ds.append(D)
    return Ds

if __name__=="__main__":
    ntests = 200
    n = 70
    approx_facts = [ 0.8 + 0.05*i for i in range(9) ]
    g6k = gen_cvpp_g6k(55,betamax=None,k=None,bits=11.705)
    # g6k = gen_cvpp_g6k(n,betamax=50,k=None,bits=24.705)
    # g6k.dump_on_disk(f"cvppg6k_n{n}_{hex(randrange(2**12))[2:]}.pkl")
    Ds = run_exp(g6k,ntests,approx_facts)

    print(Ds)
