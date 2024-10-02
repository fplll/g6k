from experiments.lwe_gen import *

import sys,os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from math import log, sqrt

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

try:
  from g6k import Siever, SieverParams
  from g6k.algorithms.bkz import pump_n_jump_bkz_tour
  from g6k.utils.stats import dummy_tracer
except ImportError:
  raise ImportError("g6k not installed")

from LatticeReduction import LatticeReduction, BKZ_SIEVING_CROSSOVER

import pickle
MAX_LOOPS = 2
inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"

def flatter_interface( fpylllB ):
    flatter_is_installed = os.system( "flatter -h > /dev/null" ) == 0

    if flatter_is_installed:
        basis = '[' + fpylllB.__str__() + ']'
        seed = randrange(2**32)
        filename = f"lat{seed}.txt"
        filename_out = f"redlat{seed}.txt"
        with open(filename, 'w') as file:
            file.write( "["+fpylllB.__str__()+"]" )

        out = os.system( "flatter " + filename + " > " + filename_out )
        time.sleep(float(0.05))
        os.remove( filename )

        B = IntegerMatrix.from_file( filename_out )
        os.remove( filename_out )
    else:
        print("Flatter issues")
    return B

def kyberGen(n, q = 3329, eta = 3, k=1):
    polys = []
    for i in range(k*k):
        polys.append( uniform_vec(n,0,q) )
    A = module(polys, k, k)

    return A,q

def se_gen(k,n,eta):
    s = binomial_vec(k*n, eta)
    e = binomial_vec(k*n, eta)
    return s, e

def generateLWEInstances(n, q = 3329, eta = 3, k=1, ntar=5):
    A,q = kyberGen(n,q = q, eta = eta, k=k)
    bse = []
    for _ in range(ntar):
        s, e = se_gen(k,n,eta)
        b = (s.dot(A) + e) % q
        bse.append( (b,s,e) )

    return A,q,bse

def gen_and_dump_lwe(n, q, eta, k, ntar, seed=0):
    print(f"- - - n,k,seed={n,k,seed} - - - gen")
    A,q,bse= generateLWEInstances(n, q, eta, k, ntar)

    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "wb") as fl:
        pickle.dump({"A": A, "q": q, "eta": eta, "k": k, "bse": bse}, fl)

def load_lwe(n,q,eta,k,seed=0):
    print(f"- - - n,k,seed={n,k,seed} - - - load")
    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_

def prepare_kyber(n,q,eta,k,betapre,seed=[0,0], nthreads=5): #for debug purposes
    """
    Prepares a kyber instance. Attempts to call load_lwe to load instances, then extracts
    the instanse bse[seed[1]]. If load fails, calls gen_and_dump_lwe. Then it attenmnpts to load
    an already preperocessed lattice. If fails, it preprocesses one.
    """
    report = {
        "kyb": ( n,q,eta,k,seed ),
        "beta": betapre,
        "time": 0
    }

    try: #try load lwe instance
        A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    except FileNotFoundError: #if no such, create one
        gen_and_dump_lwe(n, q, eta, k, 5, seed[0]) #ntar = 5
        A, q, eta,k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    #try load reduced kyber
    try:
        with open(out_path + f"kyb_preprimal_{n}_{q}_{eta}_{k}_{seed[0]}_{betapre}.pkl", "rb") as file:
            B = pickle.load(file)
    except (FileNotFoundError, EOFError): #if no such, create one
        B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
        for i in range( k*n ):
            B[i][i] = int( q )
        for i in range(k*n, 2*k*n):
            B[i][i] = 1
        for i in range(k*n, 2*k*n):
            for j in range(k*n):
                B[i][j] = int( A[i-k*n,j] )

        B = IntegerMatrix.from_matrix( B )
        #nthreads=5 by default since preprocessing operates with small blocksizes
        LR = LatticeReduction( B,threads_bkz=nthreads )
        for beta in range(4,betapre+1):
            then = time.perf_counter()
            LR.BKZ( beta )
            round_time = time.perf_counter()-then
            print(f"Preprocess BKZ-{beta} done in {round_time}")
            report["time"] += round_time

        with open(out_path + f"kyb_preprimal_{n}_{q}_{eta}_{k}_{seed[0]}_{betapre}.pkl", "wb") as file:
            pickle.dump( LR.basis, file )
        B = LR.basis
        with open(out_path + f"report_pre_{n}_{q}_{eta}_{k}_{seed[0]}_{betapre}.pkl", "wb") as file:
            pickle.dump( report, file )

    return B, A, q, eta,k, bse

def attack_on_kyber(n,q,eta,k,betapre,betamax,ntours=5,seed=[0,0],nthreads=5):
    # prepeare the lattice
    print( f"launching {n,q,eta,k,seed}" )
    # A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    B, A, q, eta,k, bse = prepare_kyber(n,q,eta,k,betapre,seed, nthreads=5)
    dim = B.nrows+1 #dimension of Kannan

    print(f"Total instances per lat: {len(bse)} seed={seed[1]}")
    b, s, e = bse[seed[1]]

    r,c = A.shape
    print(f"Shape: {A.shape}, n, k: {n,k}")
    t = np.concatenate([b,[0]*r]) #BDD target
    x = np.concatenate([b-e,s,[-1]]) #BBD solution
    sol = np.concatenate([e,-s,[1]])

    B = [ [ bb for bb in b ]+[0] for b in B ] + [ (dim-1)*[0] + [1] ]

    for j in range(k*n):
        B[-1][j] = int( t[j] )
    C = IntegerMatrix.from_matrix( B )
    B = np.array( B )
    tarnrmsq = 1.01*(sol.dot(sol))

    G = GSO.Mat(C,float_type="dd", U=IntegerMatrix.identity(dim,int_type=C.int_type), UinvT=IntegerMatrix.identity(dim,int_type=C.int_type))
    G.update_gso()

    print(G.get_r(0,0)**0.5)

    report = {
        "kyb": ( n,q,eta,k ),
        "beta": 2,
        "time": 0,
        "projinfo": {}
    }
    lll = LLL.Reduction(G)
    then = time.perf_counter()
    lll()
    llltime = time.perf_counter() - then
    report = {
        "kyb": ( n,q,eta,k ),
        "beta": 2,
        "time": llltime,
        "projinfo": {}
    }
    if lll.M.get_r(0,0) <= tarnrmsq:
        print(f"LLL recovered secret!")
        report["beta"] = beta
        return report

    flags = BKZ.AUTO_ABORT|BKZ.MAX_LOOPS|BKZ.GH_BND
    bkz = BKZReduction(G)

    cumtime = 0
    for beta in range(betapre-1,min(betamax+1,BKZ_SIEVING_CROSSOVER)):    #BKZ reduce the basis
        par = BKZ.Param(beta,
                               max_loops=MAX_LOOPS,
                               flags=flags,
                               strategies=BKZ.DEFAULT_STRATEGY
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        curnrm = np.array( bkz.M.B[0] ).dot( np.array( bkz.M.B[0] ) )**(0.5)
        # print(f"BKZ-{beta} done in {round_time} | {curnrm}")
        slope = basis_quality(bkz.M)["/"]
        print(f"Enum beta: {beta:}, done in: {round_time : 0.4f}, slope: {slope}  log r00: {log( bkz.M.get_r(0,0),2 )/2 : 0.5f} task_id = {seed}")
        report["time"] += round_time

        # ind, projfact, projsec = throw_vec( bkz.M, sol, beta ) #get info on the last projective lattice that contains a short projection
        # report["projinfo"][beta] = { "i": ind, "projfact": projfact, "projsec": projsec }

        if bkz.M.get_r(0,0) <= tarnrmsq:
            print(f"succsess! beta={beta}")
            report["beta"] = beta
            return report

    M = bkz.M
    try:
        param_sieve = SieverParams()
        param_sieve['threads'] = nthreads #10
        param_sieve['default_sieve'] = "bgj1" #"bgj1" "bdgl2"
        g6k = Siever(M, param_sieve)

        #we do not use LatticeReduction here since we do not neccesarily
        #want to run all the tours and can interupt after any given one.
        for beta in range(max(BKZ_SIEVING_CROSSOVER,betapre-1),betamax+1):
            for t in range(MAX_LOOPS):
                then_round=time.perf_counter()
                pump_n_jump_bkz_tour(g6k, dummy_tracer, beta, jump=1,
                 dim4free_fun="default_dim4free_fun",
                 extra_dim4free=0,
                 pump_params={'down_sieve': False},)
                round_time = time.perf_counter()-then_round

                # print('tour ', t, ' beta:',beta,' done in:', round_time, 'slope:', basis_quality(M)["/"], 'log r00:', float( log( g6k.M.get_r(0,0),2 )/2 ), 'task_id = ', seed)
                slope = basis_quality(M)["/"]
                print(f"Sieve tour: {t}, beta: {beta:}, done in: {round_time : 0.4f}, slope: {slope : 0.6f}, log r00: {log( g6k.M.get_r(0,0),2 )/2 : 0.5f} task_id = {seed}")
                sys.stdout.flush()  #flush after the BKZ call

                report["time"] += round_time

                if M.get_r(0,0) <= tarnrmsq:
                    print(f"succsess! beta={beta}")
                    report["beta"] = beta
                    return report
    except Exception as excpt:
        print( excpt )
        print("Sieving died!")
        pass

    return report
if __name__ == "__main__":
    path = "exp_folder/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

    nthreads = 2
    nworkers = 2
    lats_per_dim = 2
    inst_per_lat = 5 #how many instances per A, q
    q, eta = 3329, 3
    nks = [ (108+10*i,1) for i in range(1) ]
    betapre,betamax = 38, 75

    output = []
    pool = Pool( processes = nworkers )
    tasks = []

    RECOMPUTE_INSTANCE = False
    RECOMPUTE_KYBER = True
    if RECOMPUTE_INSTANCE:
        for nk in nks:
            n, k = nk[0], 1
            for latnum in range(lats_per_dim):
                gen_and_dump_lwe(nk[0], q, eta,k, ntar=inst_per_lat, seed=latnum)
    if RECOMPUTE_KYBER or RECOMPUTE_INSTANCE:
        pretasks = []
        for nk in nks:
            n, k = nk[0], 1
            for latnum in range(lats_per_dim):
                pretasks.append( pool.apply_async(
                prepare_kyber, (n,q,eta,k,betapre,[latnum,0], nthreads)
                ) )

        for t in pretasks:
            t.get()

    for nk in nks:
        n, k = nk[0], 1
        for latnum in range(lats_per_dim):
            for tstnum in range(inst_per_lat):
                # output.append( attack_on_kyber(nk[0],q,eta,k,57,70,5,[latnum,tstnum],nthreads) )
                tasks.append( pool.apply_async(
                    attack_on_kyber, (nk[0],q,eta,k,betapre,betamax,5,[latnum,tstnum],nthreads)
                    ) )


    for t in tasks:
            output.append( t.get() )

    pool.close()

    name = f"exp105-2.pkl"
    with open( path+name, "wb" ) as file:
        pickle.dump( output,file )

    print(f"- - - output - - -")
    print(output)

    time.sleep(0.5)
