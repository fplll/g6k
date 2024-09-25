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

def reduce_to_fund_par_proj(B_gs,t_gs,dim):
    t_gs_save = deepcopy( t_gs )
    c = [0 for i in range(dim)]
    for i in range(dim):
        for j in range(i,-1,-1):
            mu = round( t_gs[j] / B_gs[j][j] )
            t_gs -= B_gs[j] * mu
            c[j] -= mu
    for i in range(dim):
        t_gs_save += c[i] * B_gs[i]
    return t_gs_save


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

def random_on_sphere(d,r):
    """
    d - dimension of vector
    r - radius of the sphere
    """
    u = np.random.normal(0,1,d)  # an array of d normally distributed random variables
    d=np.sum(u**2) **(0.5)
    return r*u/d
