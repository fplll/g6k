from lwe_gen import *

import os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction

def flatter_interface( fpylllB ):
    flatter_is_installed = os.system( "flatter -h flatter -h > /dev/null" ) == 0

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
        print("aaa")
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
    A,q,bse= generateLWEInstances(n, q, eta, k, ntar)

    with open(f"lwe_instance_{n}_{q}_{eta}_{seed}", "wb") as fl:
        pickle.dump({"A": A, "q": q, "eta": eta, "bse": bse}, fl)

def load_lwe(n,q,eta,seed=0):
    with open(f"lwe_instance_{n}_{q}_{eta}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, bse_ = D["A"], D["q"], D["eta"], D["bse"]
    return A_, q_, eta_, bse_

def attack_on_kyber(n,q,eta,betamax,ntours=5, seed=[0,0]):
    # seed = #lat, #vec
    # prepeare the lattice
    print( f"launching {n,q,eta,k}" )

    try:
        A, q, eta, bse = load_lwe(n,q,eta,seed[0]) #D["A"], D["q"], D["bse"]
    except FileNotFoundError:
        gen_and_dump_lwe(n, q, eta, k, ntar, seed[0])
        A, q, eta, bse = load_lwe(n,q,eta,seed[0]) #D["A"], D["q"], D["bse"]

    b, s, e = bse[seed[1]]
    r,c = A.shape
    t = np.concatenate([b,[0]*r]) #BDD target
    x = vector( np.concatenate([b-e,s,[-1]]) ) #BBD solution
    sol = vector( np.concatenate([e,-s,[1]]) )

    B = matrix( 2*k*n+1, 2*k*n+1 )
    for i in range( k*n ):
        B[i,i] = q
    for i in range(k*n, 2*k*n+1):
        B[i,i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i,j] = A[i-k*n,j]

    for j in range(k*n):
        B[-1,j] = t[j]

    tarnrmsq = ( 1.01 * norm(sol).n() )**2

    then = perf_counter()
    C = flatter_interface(B)
    print(f"flatter done in {perf_counter()-then}")

    G = GSO.Mat(C,float_type="dd")
    G.update_gso()

    print(G.get_r(0,0)^0.5)

    lll = LLL.Reduction(G)
    lll()

    flags = BKZ.AUTO_ABORT|BKZ.MAX_LOOPS|BKZ.GH_BND
    bkz = BKZReduction(G)

    report = {
        "kyb": ( n,q,eta,k ),
        "beta": 0,
        "time": 0,
        "projinfo": {}
    }
    cumtime = 0
    for beta in range(2,min(56,betamax+1)):    #BKZ reduce the basis
        par = BKZ.Param(beta,
                               max_loops=5,
                               flags=flags,
                               strategies=BKZ.DEFAULT_STRATEGY
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time} | {norm(vector(bkz.M.B[0])).n()}")
        report["time"] += round_time

        ind, projfact, projsec = throw_vec( bkz.M, sol, beta )
        report["projinfo"][beta] = { "i": ind, "projfact": projfact, "projsec": projsec }

        if bkz.M.get_r(0,0) <= tarnrmsq:
            print(f"succsess!")
            report["beta"] = beta
            return report
    M = bkz.M
    param_sieve = SieverParams()
    param_sieve['threads'] = 10
    param_sieve['default_sieve'] = "bgj1" #"bgj1"
    g6k = Siever(M, param_sieve)

    for beta in range(55,betamax+1):
        for t in range(global_variables.bkz_max_loops):
            then_round=time.perf_counter()
            my_pump_n_jump_bkz_tour(g6k, dummy_tracer, beta, jump=1,
             filename="devnull", seed=1,
             dim4free_fun="default_dim4free_fun",
             extra_dim4free=0,
             pump_params={'down_sieve': False},
             verbose=verbose)
            round_time = time.perf_counter()-then_round

            print('tour ', t, ' bkz for beta=',blocksize,' done in:', round_time, 'slope:', basis_quality(M)["/"], 'log r00:', float( log( g6k.M.get_r(0,0),2 )/2 ), 'task_id = ', task_id)
            sys.stdout.flush()  #flush after the BKZ call

            report["time"] += round_time

            ind, projfact, projsec = throw_vec( bkz.M, sol, beta )
            report["projinfo"][beta] = { "i": ind, "projfact": projfact, "projsec": projsec }

            if M.get_r(0,0) <= tarnrmsq:
                print(f"succsess!")
                report["beta"] = beta
                return report

    return report

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle

path = "exp_folder/"
isExist = os.path.exists(path)
if not isExist:
    try:
        os.makedirs(path)
    except:
        pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

nthreads = 10
lats_per_dim = 2
inst_per_lat = 2 #how many instances per A, q
q, eta, k, ntar

nks = [ (100+10*i,1) for i in range(2) ]
eta = 3
output = []
pool = Pool(processes = nthreads )
tasks = []

for nk in nks:
    for latnum in range(lats_per_dim):
        gen_and_dump_lwe(n, q, eta, k, ntar, latnum)
        # def attack_on_kyber(n,q,eta,betamax,ntours=5, seed=[0,0])
        for tstnum in range(inst_per_lat)
            tasks.append( pool.apply_async(
                attack_on_kyber, (nk[0],q,eta,65,5,[latnum,tstnum])
                ) )

for t in tasks:
        output.append( t.get() )

pool.close()
print( output )

name = f"exp{randrange(1000)}.pkl"
with open( path+name, "wb" ) as file:
    pickle.dump( output,file )
