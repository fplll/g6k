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

n = 14
B = IntegerMatrix(n,n)
B.randomize("qary", k=n//2, bits=11.705)
ft = "ld" if n<193 else "dd"
G = GSO.Mat(B, float_type=ft)
G.update_gso()

lll = LLL.Reduction(G)
lll()
int_type = G.B.int_type
G = GSO.Mat( G.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
G.update_gso()

param_sieve = SieverParams()
param_sieve['threads'] = 5
# param_sieve['db_size_factor'] = 3.75
param_sieve['default_sieve'] = "bdgl2" #"bgj1"
g6k = Siever(G,param_sieve)
g6k.initialize_local(5,5,n)
g6k()

filename = "testpkl.pkl"
g6k.dump_on_disk(filename)
print("Dump succsess")
print(f"len g6k: {len(g6k)}")

g6k_ = Siever.restore_from_file(filename)
print(f"len g6k_: {len(g6k_)}")

l0, l1 = [tmp for tmp in g6k.itervalues()], [tmp for tmp in g6k_.itervalues()]
for i in range(len(g6k)):
    c0, c1 = np.array(l0[i]), np.array(l1[i])
    assert ( all( c0==c1 ) ), f"DB corrupted!"
