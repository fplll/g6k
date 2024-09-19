from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys

if __name__ == "__main__":

    FPLLL.set_precision(250)
    n, betamax, sieve_dim = 112, 52, 52 #n=170 is liikely to fail
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    # - - - try load a lattice - - -
    filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
    nothing_to_load = True
    try:
        g6k = Siever.restore_from_file(filename)
        G = g6k.M
        B = G.B
        nothing_to_load = False
        print(f"Load seems to succseed...")
    except Exception as excpt:
        print(excpt)
        pass
    # - - - end try load a lattice - - -

    # - - - Make all fpylll objects - - -
    if nothing_to_load:
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
    # - - - end Make all fpylll objects - - -

    c = [ randrange(-2,3) for j in range(n) ]
    e = np.array( [ randrange(-8,8) for j in range(n) ],dtype=np.int64 )

    print(f"gauss: {gaussian_heuristic(G.r())**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

    e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
    print(f"projected (e_@e_): {(e_@e_)} vs r/4: {G.get_r(n-sieve_dim, n-sieve_dim)/4}")
    print("projected target squared length:", 1.01*(e_@e_))

    b = G.B.multiply_left( c )
    b_ = np.array(b,dtype=np.int64)
    t_ = e+b_
    t = [ int(tt) for tt in t_ ]

    param_sieve = SieverParams()
    param_sieve['threads'] = 2
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")

    #assert(False)

    t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
    print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")
    #retrieve the projective sublattice
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
    t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
    t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer


    # - - - prelim check - - -
    out = to_canonical_scaled( G,t_gs_reduced,offset=sieve_dim )

    projerr = G.to_canonical( G.from_canonical(e,start=n-sieve_dim), start=n-sieve_dim)
    diff_v =  np.array(projerr, dtype=np.float64)-np.array(out, dtype=np.float64)
    # print(f"Diff btw. cvp and slicer: {diff_v}")

    N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
    N.update_gso()
    bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
    bab_0 = N.babai(tmp)

    bab_01=np.array( bab_0+bab_1 )
    print((f"recovered*B^(-1): {bab_0+bab_1}"))
    print(c)
    print(f"Coeffs of b found: {(c==bab_01)}")
    succ = all(c==bab_01)
    print(f"Babai Succsess: {succ}")
    # - - - end prelim check - - -
    # - - - extra check - - -
    bab_t = np.array( g6k.M.babai(t) )
    print(f"Coeffs of b found: {(c==bab_t)}")
    succ = all(c==bab_t)
    print(f"Babai Succsess: {succ}")

    # - - - end extra check - - -

    if not succ:
        filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
        g6k.dump_on_disk( filename )
        #then = perf_counter()

        #out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000)
        slicer = RandomizedSlicer(g6k)
        slicer.set_nthreads(2);

        print("target:", [float(tt) for tt in t_gs_reduced])
        print("dbsize", g6k.db_size())

        slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=900)

        blocks = 2 # should be the same as in siever
        blocks = min(3, max(1, blocks))
        blocks = min(int(sieve_dim / 28), blocks)
        sp = SieverParams()
        N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
        buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
        buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
        buckets = max(buckets, 2**(blocks-1))

        print("blocks: ", blocks, " buckets: ", buckets )
        # e_ = np.array( from_canonical_scaled(g6k.M,e,offset=sieve_dim) )

        # print(f"(e_@e_): {(e_@e_)} vs r: {g6k.M.get_r(n-sieve_dim, n-sieve_dim)}")
        # print("target length:", 1.01*(e_@e_))
        #slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (1.01*(e_@e_)))
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (1.01*(e_@e_)))
        iterator = slicer.itervalues_t()
        for tmp in iterator:
            out_gs_reduced = tmp  #cdb[0]
            break
        out_gs = out_gs_reduced + t_gs_shift

        # - - - Check - - - -
        out = to_canonical_scaled( G,out_gs,offset=sieve_dim )

        projerr = G.to_canonical( G.from_canonical(e,start=n-sieve_dim), start=n-sieve_dim)
        diff_v =  np.array(projerr)-np.array(out)
        # print(f"Diff btw. cvp and slicer: {diff_v}")

        N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
        N.update_gso()
        bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
        tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
        tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
        bab_0 = N.babai(tmp)

        bab_01=np.array( bab_0+bab_1 )
        #print((f"recovered*B^(-1): {bab_0+bab_1}"))
        print(c)
        #print(f"Coeffs of b found: {(c==bab_01)}")
        print(f"Succsess: {all(c==bab_01)}")

