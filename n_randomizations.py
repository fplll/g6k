from fpylll import *
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import argparse
import sys

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

def run_exp(lat_id, n, betamax, sieve_dim, range_, Nexperiments):
    babai_suc = 0
    approx_fact = 1.1
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    print(f"launching n, betamax, sieve_dim = {n, betamax, sieve_dim}")
    print(f"range_: {range_}")

    slicer_suc = [0]*len(range_)
    slicer_fail = [0]*len(range_)
    # - - - try load a lattice - - -
    filename = f"saved_lattices/bdgl2_n{n}_b{sieve_dim}_{lat_id}.pkl"
    nothing_to_load = True
    try:
        g6k = Siever.restore_from_file(filename)
        G = g6k.M
        B = G.B
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



    param_sieve = SieverParams()
    param_sieve['threads'] = 1
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

    for i in range(Nexperiments):

        print("Running experiment ", i, "out of ", Nexperiments-1)

        c = [ randrange(-10,10) for j in range(n) ]
        #e = np.array( [ randrange(-8,9) for j in range(n) ],dtype=np.int64 )
        e = np.array( random_on_sphere(n, 1.0*gh/2) )

        print(f"gauss: {gh} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")
        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
        print("projected target squared length:", 1.01*(e_@e_))

        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t_ = e+b_
        t = [ int(tt) for tt in t_ ]

        t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
        #print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")
        #retrieve the projective sublattice
        B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
        t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer


        # - - - Babai check - - -
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
        #print((f"recovered*B^(-1): {bab_0+bab_1}"))
        #print(c)
        #print(f"Coeffs of b found: {(c==bab_01)}")
        succ = all(c==bab_01)
        print(f"Babai Success: {succ}")
        if succ:
            babai_suc+=1


        if not succ:

            #filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
            #g6k.dump_on_disk( filename )
            #then = perf_counter()
            ctr = 0
            this_instance_succseeded = False
            for nrand in range_:
                if this_instance_succseeded: #can only enter here after a succsessful slicer
                    slicer_suc[ctr] += 1
                    continue

                slicer = RandomizedSlicer(g6k)
                slicer.set_nthreads(2);
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)
                try:
                    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (approx_fact*(e_@e_)))
                    iterator = slicer.itervalues_t()
                    for tmp in iterator:
                        out_gs_reduced = tmp  #cdb[0]
                        break
                    out_gs = out_gs_reduced + t_gs_shift

                    # - - - Check - - - -
                    out = to_canonical_scaled( G,out_gs,offset=sieve_dim )

                    projerr = G.to_canonical( G.from_canonical(e,start=n-sieve_dim), start=n-sieve_dim)
                    diff_v =  np.array(projerr)-np.array(out)


                    N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
                    N.update_gso()
                    bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
                    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
                    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
                    bab_0 = N.babai(tmp)

                    bab_01=np.array( bab_0+bab_1 )
                    print(f"Success: {all(c==bab_01)}")
                    if (all(c==bab_01)):
                        slicer_suc[ctr] += 1
                        this_instance_succseeded = True
                    else:
                        slicer_fail[ctr] += 1


                except Exception as e: print(e)
                ctr+=1
    print(f"Lattice-{lat_id} processed...")
    print(babai_suc)
    print(slicer_suc)
    print(slicer_fail)

    density_plot = []
    cntr = 0
    for nrand in range_:
        density_plot.append( (nrand,slicer_suc[cntr]+babai_suc) )
        cntr+=1
    return density_plot

#paramset1 = {"n": 110, "b": [i for i in range(42, 56)], "nrands": [i for i in range(600,900,50)] }
#paramset2 = {"n": 120, "b": [i for i in range(42, 56)], "nrands": [i for i in range(600,900,50)] }

if __name__ == '__main__':
    n_rerand_min, n_rerand_max, step = 10, 71, 10
    range_ = range(n_rerand_min, n_rerand_max, step)
    # babai_suc = 0
    # slicer_suc = [0]*len(range_)
    # slicer_fail = [0]*len(range_)
    Nexperiments = 10
    Nlats = 5
    path = "saved_lattices/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass


    FPLLL.set_precision(250)
    n, betamax, sieve_dim = 50, 48, 50
    nthreads = 2
    pool = Pool(processes = nthreads )
    tasks = []

    density_plots = []
    for lat_id in range(Nlats):
        # density_plot = run_exp(lat_id, n, betamax, sieve_dim, range_, Nexperiments)
        # density_plots.append(density_plot)
        tasks.append( pool.apply_async(
            run_exp, (lat_id, n, betamax, sieve_dim, range_, Nexperiments)
        ) )

    for t in tasks:
        density_plots.append( t.get() )


    with open(f"nrand_{n}_exp.pkl", "wb") as file:
        pickle.dump( density_plots, file )

    print(density_plots)
    print(Nexperiments)
