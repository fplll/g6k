from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys

import time, pickle
from random import shuffle

from hyb_att_on_kyber import alg_2_batched

# n, betamax, sieve_dim = 140, 45, 45 #n=170 is liikely to fail
n, betamax, sieve_dim = 100, 48, 50 #n=170 is liikely to fail

bits=11.705
ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

loadsucc = False

try:
    with open(f"qary_{n}_{betamax}_{bits:.4f}.pkl", "rb") as file:
        B = pickle.load( file )
    loadsucc = True
except FileNotFoundError:
    print(f"Nothing to load. Computing")
    pass

if not loadsucc:
    B = IntegerMatrix(n,n)
    B.randomize("qary", k=n//2, bits=bits)
    G = GSO.Mat(B, float_type=ft)
    G.update_gso()

    if sieve_dim<30: print("Slicer is not implemented on dim < 30")
    if sieve_dim<40: print("LSH won't work on dim < 40")

    lll = LLL.Reduction(G)
    lll()

    bkz = LatticeReduction(B)
    for beta in range(5,betamax+1):
        then_round=time.perf_counter()
        bkz.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
        sys.stdout.flush()

    with open(f"qary_{n}_{betamax}_{bits:.4f}.pkl", "wb") as file:
        pickle.dump(bkz.basis, file)
    B = bkz.gso.B

int_type = B.int_type
G = GSO.Mat(B , U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
G.update_gso()
lll = LLL.Reduction( G )
lll()

gh = gaussian_heuristic(G.r())**0.5
param_sieve = SieverParams()
param_sieve['threads'] = 4
g6k = Siever(G,param_sieve)
g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
print("Running bdgl2...")
g6k(alg="bdgl2")
g6k.M.update_gso()

print(f"dbsize: {len(g6k)}")
time.sleep(0.2)

c = [ randrange(-30,31) for j in range(n) ]
e = np.array( random_on_sphere(n,(1/14.)*gh), dtype=np.float64 )
b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ float(tt) for tt in t_ ]
e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) , dtype=np.float64 )
egs_ = np.array( G.from_canonical(e)[n-sieve_dim:], dtype=np.float64 )
egs_ = np.array( G.to_canonical(egs_,start=n-sieve_dim), dtype=np.float64 )
print(f"ee_: {(e_@e_)}")
print(f"egs_: {(egs_@egs_)}")
print(f"rii: {G.r()[-sieve_dim]}")
print(f"r: {[rr**0.5 for rr in G.r()]}")

target_candidates = [t]
for _ in range(0):
    e2 = np.array( random_on_sphere(n,14.2*gh), dtype=np.float64 ) #np.array( [ randrange(0,1) for j in range(n) ],dtype=np.int64 )
    tcand_ = e2 + e + b_
    tcand = [ int(tt) for tt in t_ ]
    target_candidates.append( tcand )
shuffle(target_candidates)

#alg_2_batched( g6k,target_candidates,H11, nthreads=1, tracer_alg2=None )
bab_01 = np.array( alg_2_batched( g6k,target_candidates,dist_sq_bnd=e_@e_  ) )
print(f"e_: {e_}")
print(f"c: {c}")
print(f"bab01:{bab_01}")
print(f"alg_2_batch succsess: {(bab_01==c)}")

lol = np.array( G.babai(t) )
print(f"babai succsess: {(lol==c)}")
