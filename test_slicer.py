from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import Randomized_slicer
from utils import *

if __name__ == "__main__":
    n, betamax, sieve_dim = 50, 35, 50
    B = IntegerMatrix(n,n)
    B.randomize("qary", k=n//2, bits=11.705)
    ft = "ld" if n<193 else "dd"
    G = GSO.Mat(B, float_type=ft)
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
    c = [ randrange(-3,4) for j in range(n) ]
    e = np.array( [ randrange(-2,3) for j in range(n) ],dtype=np.int64 )
    print(f"gauss: {gaussian_heuristic(G.r())**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

    b = G.B.multiply_left( c )
    b_ = np.array(b,dtype=np.int64)
    t_ = e+b_
    t = [ int(tt) for tt in t_ ]

    param_sieve = SieverParams()
    param_sieve['threads'] = 5
    # param_sieve['db_size_factor'] = 3.75
    param_sieve['default_sieve'] = "bgj1"
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    g6k()
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")

    t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
    print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")

    then = perf_counter()

    #out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000)
    slicer = Randomized_slicer()

    slicer.grow_db_with_target([float(tt) for tt in t_gs], n_per_target=100)
