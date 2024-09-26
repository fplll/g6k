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

def run_exp(lat_id, n, betamax, sieve_dim, range_, Nexperiments, nthreads=1):
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

        bkz = LatticeReduction(B, threads_bkz=nthreads)
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


    # rinv_ = np.array( [sqrt(gh/tt) for tt in G.r()[G.d-sieve_dim:]], dtype=np.float64 ) #transform. coeffs btwn scaled and non-scaled gs coords
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

    N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
    N.update_gso()

    for i in range(Nexperiments):

        print("Running experiment ", i, "out of ", Nexperiments-1)
        if i%5==0:
            sys.stdout.flush()

        c = [ randrange(-10,10) for j in range(n) ]
        #e = np.array( [ randrange(-8,9) for j in range(n) ],dtype=np.int64 )
        e = np.array( random_on_sphere(n, 0.49*gh) )

        print(f"gauss: {gh} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")
        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
        print("projected target squared length:", 1.01*(e_@e_))

        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t_ = e+b_
        t = [ int(tt) for tt in t_ ]

        #size red
        t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

        #Checked for t_gs_reduced!
        t_gs_non_scaled = G.from_canonical(t)[-sieve_dim:]
        shift_babai_c = G.babai((n-sieve_dim)*[0] + list(t_gs_non_scaled), start=n-sieve_dim,gso=True)
        shift_babai = G.B.multiply_left( (n-sieve_dim)*[0] + list( shift_babai_c ) )
        t_gs_reduced = from_canonical_scaled( G,np.array(t)-shift_babai,offset=sieve_dim ) #this is the actual reduced target
        t_gs_shift = from_canonical_scaled( G,shift_babai,offset=sieve_dim )
        print(f"nrm t_gs_reduced: {t_gs_reduced@t_gs_reduced}")
        print(f"nrm t_gs_shift: {t_gs_shift@t_gs_shift}")

        print(f"norm: {t_gs_reduced@ t_gs_reduced}")

        out = t_gs_reduced
        N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
        N.update_gso()
        bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s

        # tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
        # tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
        # bab_0 = N.babai(tmp)
        #
        # bab_01=np.array( bab_0+bab_1 )
        # succ = all(c==bab_01)

        #print((f"recovered*B^(-1): {bab_0+bab_1}"))
        #print(c)
        #print(f"Coeffs of b found: {(c==bab_01)}")

        succ = all( np.array( c[G.d-sieve_dim:] )==bab_1 )
        print(f"Babai Success: {succ}")
        if succ:
            babai_suc+=1
        if not succ:
            ctr = 0
            #this_instance_succseeded = False
            for nrand in range_:
                #if this_instance_succseeded: #can only enter here after a succsessful slicer
                #    slicer_suc[ctr] += 1
                #    continue

                slicer = RandomizedSlicer(g6k)
                slicer.set_nthreads(2);
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)
                try:
                    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (approx_fact*approx_fact*(e_@e_)))

                    iterator = slicer.itervalues_t()
                    for tmp in iterator:
                        out_gs_reduced = tmp  #cdb[0]
                        break
                    out_gs =  out_gs_reduced + t_gs_shift #t_gs_shift + suspected error

                    # - - - Check - - - -
                    out = to_canonical_scaled( G,out_gs,offset=sieve_dim )

                    # N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
                    # N.update_gso()

                    """
                    out_gs_fpylll_format = out_gs * rinv_ #translate from unsceled to the scaled representation for babai
                    bab_1 = G.babai(t_gs_non_scaled-out_gs_fpylll_format,start=n-sieve_dim) #last sieve_dim coordinates of s
                    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
                    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
                    bab_0 = N.babai(tmp)
                    """
                    bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
                    tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
                    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
                    bab_0 = N.babai(tmp)


                    bab_01=np.array( bab_0+bab_1 ) #shifted answer. Good since it is smaller, thus less rounding error
                    bab_01 += np.array(shift_babai_c)
                    print(f"Slicer Success: {all(c==bab_01)}")
                    out_gs_reduced = np.array( out_gs_reduced )
                    found_nrm_sq = out_gs_reduced@out_gs_reduced
                    if (all(c==bab_01)):
                        print(f"SUCCESS")
                        print(f"found found_nrm_sq (succ): {found_nrm_sq}")
                        slicer_suc[ctr] += 1
                       # this_instance_succseeded = True
                    else:
                        print("SLICER fail")
                        slicer_fail[ctr] += 1
                        print(f"found found_nrm_sq: {found_nrm_sq}")
                        # print(f"c: {c}")
                        # print(f"bab_01: {bab_01}")
                        # print(f"c-shift_c{np.array(c)-np.array(shift_babai_c)}")


                except Exception as e: raise e #print(e)
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
    n_rerand_min, n_rerand_max, step = 20, 171, 50
    range_ = range(n_rerand_min, n_rerand_max, step)
    # babai_suc = 0
    # slicer_suc = [0]*len(range_)
    # slicer_fail = [0]*len(range_)
    Nexperiments = 200
    Nlats = 5
    path = "saved_lattices/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass
    #TODO: make the factor in front of gh as input
    # TODO: rename approx_fact as len_slack
    # Output on
    #  Nexperiments = 200
    # Nlats = 5
    # n, betamax, sieve_dim = 60, 48, 60
    # 0.49*gh
    #
    # [[(20, 118), (70, 197), (120, 199), (170, 200)], [(20, 124), (70, 196), (120, 200), (170, 200)], [(20, 121), (70, 195), (120, 200), (170, 200)], [(20, 123), (70, 194), (120, 199), (170, 200)], [(20, 120), (70, 195), (120, 199), (170, 200)]]
    FPLLL.set_precision(250)

    n, betamax, sieve_dim = 60, 48, 60
    nthreads = 5
    slicer_threads = 2
    pool = Pool(processes = nthreads )
    tasks = []

    density_plots = []
    for lat_id in range(Nlats):
        # density_plot = run_exp(lat_id, n, betamax, sieve_dim, range_, Nexperiments)
        # density_plots.append(density_plot)
        tasks.append( pool.apply_async(
            run_exp, (lat_id, n, betamax, sieve_dim, range_, Nexperiments, slicer_threads)
        ) )

    for t in tasks:
        density_plots.append( t.get() )


    with open(f"nrand_{n}_exp.pkl", "wb") as file:
        pickle.dump( density_plots, file )

    print(density_plots)
    print(Nexperiments)
