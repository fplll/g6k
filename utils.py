import sys, os
import glob #for automated search in subfolders
import numpy as np
from experiments.lwe_gen import generateLWEInstance, binomial_vec, uniform_vec
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
    try:
        g6k()
    except SaturationError as saterr:
        print(saterr)
        pass
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
