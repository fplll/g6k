from __future__ import absolute_import
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever

n, k = 68, 38
A = IntegerMatrix.random(n, "qary", k=k, bits=14.02)
G = GSO.Mat( A, float_type="dd")
G.update_gso()
lll = LLL.Reduction( G )
lll()

betamax = 14
flags = BKZ.AUTO_ABORT|BKZ.MAX_LOOPS|BKZ.GH_BND
bkz = BKZReduction(G)
for beta in range(2,betamax+1):
    par = BKZ.Param(beta,
    max_loops=5,
    flags=flags,
    strategies=BKZ.DEFAULT_STRATEGY
    )
    then_round=time.perf_counter()
    bkz(par)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")

A = bkz.M.B
g6k = Siever(A)
llft, lft, rft = 8, n-45, n
g6k.initialize_local(llft, lft, rft)
g6k()
print( f"Init g6k db size: {len(g6k)}" )

for shl in range(lft-llft):
    if shl%5==0:
        print( f"g6k db size: {len(g6k)}" )
        print(f"Left context: {g6k.l}; right context: {g6k.r}")
    g6k.extend_left()
    g6k()
print( f"g6k db size: {len(g6k)}" )
print(f"Left context: {g6k.l}; right context: {g6k.r}")

db = list(g6k.itervalues())[:10]
siever_lens = [ sum([vv**2 for vv in A[llft:rft].multiply_left(c)])**0.5 for c in db ]
print( siever_lens )
basis_lens = [ sum([vv**2 for vv in v])**0.5 for v in A ]
print( basis_lens )

print(f"Min. basis vctr len {min(basis_lens)} vs. min sieve len: {min(siever_lens)}")
