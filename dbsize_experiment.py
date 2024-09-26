from fpylll import *
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import argparse
import sys
from hybrid_estimator.batchCVP import batchCVPP_cost

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

def run_exp(lat_id, n, betamax, sieve_dim, shrink_factor, n_shrinkings, Nexperiments, nthreads):
    babai_suc = 0
    approx_fact = 1.055
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    print(f"launching n, betamax, sieve_dim = {n, betamax, sieve_dim}")

    slicer_suc = [0]*n_shrinkings
    slicer_fail = [0]*n_shrinkings
    # - - - try load a lattice - - -
    filename = f"saved_lattices/bdgl2_n{n}_b{sieve_dim}_{lat_id}.pkl"
    nothing_to_load = True
    try:
        g6k = Siever.restore_from_file(filename)
        G = g6k.M
        nothing_to_load = False
        print(f"Load succeeded...")
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

        bkz = LatticeReduction(B,threads_bkz=nthreads)
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
    # make Siver object
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()
    gh = gaussian_heuristic(G.r())**0.5
    if nothing_to_load:
        g6k.dump_on_disk(filename)

    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    dbsize_start = g6k.db_size()

    for j in range(n_shrinkings):
        # slicer = RandomizedSlicer(g6k)
        # slicer.set_nthreads(nthreads);
        nrand_, _ = batchCVPP_cost(sieve_dim,100,dbsize_start**(1./sieve_dim),1) #100 can be any constant >1
        print("nrand:", (1./nrand_)**sieve_dim)
        print("Running experiment ", j, "out of ", n_shrinkings)

        for i in range(Nexperiments):
            c = [ randrange(-10,10) for k in range(n) ]
            e = np.array( random_on_sphere(n, 0.48 * gh) ) #error vector
            print(f"gauss: {gh} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")
            e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )

            b = G.B.multiply_left( c )
            b_ = np.array(b,dtype=np.int64)
            t_ = e+b_
            t = [ int(tt) for tt in t_ ]

            #project onto the last projective lattice and babai reduce
            t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
            t_gs_non_scaled = G.from_canonical(t)[-sieve_dim:]
            shift_babai_c = G.babai((n-sieve_dim)*[0] + list(t_gs_non_scaled), start=n-sieve_dim,gso=True)
            shift_babai = G.B.multiply_left( (n-sieve_dim)*[0] + list( shift_babai_c ) )
            t_gs_reduced = from_canonical_scaled( G,np.array(t)-shift_babai,offset=sieve_dim ) #this is the actual reduced target
            t_gs_shift = from_canonical_scaled( G,shift_babai,offset=sieve_dim )

            print("projected reduced target squared length:", (t_gs_reduced@t_gs_reduced))
            print("projected error squared length:", (e_@e_))


            # - - - Babai check - - -
            out = to_canonical_scaled( G,t_gs_reduced,offset=sieve_dim )
            N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
            N.update_gso()
            bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
            tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
            tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
            bab_0 = N.babai(tmp)

            bab_01=np.array( bab_0+bab_1 ) #shifted answer. Good since it is smaller, thus less rounding error
            bab_01 += np.array(shift_babai_c)
            succ = all(c==bab_01)
            print(f"Babai Success: {succ}")
            if succ:
                babai_suc+=1

            if not succ:
                #need to define it here since old targets and their rerandomizations
                #would remain to be in the db_t
                slicer = RandomizedSlicer(g6k)
                slicer.set_nthreads(nthreads);
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=ceil((1./nrand_)**sieve_dim))
                try:
                    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (approx_fact**2 * (e_@e_)))
                    iterator = slicer.itervalues_t()
                    for tmp in iterator:
                        out_gs_reduced = tmp  #cdb[0]
                        break
                    out_gs = out_gs_reduced + t_gs_shift

                    # - - - Check - - - -
                    out = to_canonical_scaled( G,out_gs,offset=sieve_dim )
                    N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
                    N.update_gso()
                    bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
                    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
                    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
                    bab_0 = N.babai(tmp)

                    bab_01 =  np.array( bab_0+bab_1 ) #shifted answer. Good since it is smaller, thus less rounding error
                    bab_01 += np.array(shift_babai_c)
                    print(f"Success: {all(c==bab_01)}")
                    if (all(c==bab_01)):
                        print(f"SUCCESS")
                        slicer_suc[j] += 1
                    else:
                        slicer_fail[j] += 1
                        print(f"FAIL")
                        found_nrm_sq = out_gs@out_gs #meant to be an error modulo Voronoi cell
                        assert ( found_nrm_sq>0.999*(e_@e_) ), f"Found impossible vector! {found_nrm_sq} < {(e_@e_)}"

                except Exception as e: print(f" - - - {e} - - -")

        g6k.shrink_db(shrink_factor*g6k.db_size())

    print(f"Lattice-{lat_id} processed...")
    print(babai_suc)
    print(slicer_suc)
    print(slicer_fail)

    density_plot = []
    cntr = 0
    s = 1
    for j in range(n_shrinkings):
        density_plot.append( (s,slicer_suc[cntr]+babai_suc) )
        cntr+=1
        s *= shrink_factor
    return density_plot


if __name__ == '__main__':

    Nexperiments = 15
    Nlats = 1
    path = "saved_lattices/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass


    FPLLL.set_precision(250)
    n, betamax, sieve_dim = 56, 45, 56
    nthreads = 2
    slicer_threads = 2
    shrink_factor = 0.95
    n_shrinkings = 20
    pool = Pool(processes = nthreads )
    tasks = []

    density_plots = []
    for lat_id in range(Nlats):
        tasks.append( pool.apply_async(
            run_exp, (lat_id, n, betamax, sieve_dim, shrink_factor, n_shrinkings, Nexperiments, slicer_threads)
        ) )

    for t in tasks:
        density_plots.append( t.get() )


    with open(f"dbsize_{n}_exp.pkl", "wb") as file:
        pickle.dump( density_plots, file )

    print(density_plots)
    print(Nexperiments)
