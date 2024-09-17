from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *

if __name__ == "__main__":
    n, betamax, sieve_dim = 55, 41, 41
    B = IntegerMatrix(n,n)
    B.randomize("qary", k=n//2, bits=11.705)
    ft = "ld" if n<193 else "dd"
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
    param_sieve['threads'] = 1
    # param_sieve['db_size_factor'] = 3.75
    #param_sieve['default_sieve'] = "bdgl1"
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")

    #assert(False)

    t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
    print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")

    #then = perf_counter()

    #out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000)
    slicer = RandomizedSlicer(g6k)

    print("target:", [float(tt) for tt in t_gs])
    print("dbsize", g6k.db_size())

    slicer.grow_db_with_target([float(tt) for tt in t_gs], n_per_target=5000)

    blocks = 2
    sp = SieverParams()
    #print(sp["db_size_factor"])
    #print(sp["db_size_base"])
    #print(sp["bdgl_bucket_size_factor"])
    #print(sp["bdgl_multi_hash"])
    #print(sp["bdgl_min_bucket_size"])


    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    e_ = np.array( from_canonical_scaled(g6k.M,e,offset=sieve_dim) )

    print("target length:", 1.01*(e_@e_))
    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (1.01*(e_@e_)))