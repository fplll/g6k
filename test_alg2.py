from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys

from hyb_att_on_kyber import alg_2_batched

n, betamax, sieve_dim = 55, 55, 50 #n=170 is liikely to fail
ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

print(f"Nothing to load. Computing")
B = IntegerMatrix(n,n)
B.randomize("qary", k=n//2, bits=11.705)
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

int_type = bkz.gso.B.int_type
G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
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

c = [ randrange(-30,31) for j in range(n) ]
e = np.array( random_on_sphere(n,0.44*gh) )
b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ int(tt) for tt in t_ ]
e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
print(f"e_: {e_}")

target_candidates = [t]
for _ in range(5):
    e2 = np.array( [ randrange(0,1) for j in range(n) ],dtype=np.int64 )
    tcand_ = e2 + e + b_
    tcand = [ int(tt) for tt in t_ ]
    target_candidates.append( tcand )

#alg_2_batched( g6k,target_candidates,H11, nthreads=1, tracer_alg2=None )
bab_01 = np.array( alg_2_batched( g6k,target_candidates,dist_sq_bnd=e_@e_  ) )
print(f"c: {c}")
print(bab_01)
print(f"alg_2_batch succsess: {(bab_01==c)}")
