import sys,os
import time
from time import perf_counter
from fpylll import *
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from utils import *

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"
#path = "saved_lattices/"
does_exist = os.path.exists(inp_path)
if not does_exist:
    sys.exit('cannot find path for input lattices')

does_exist = os.path.exists(out_path)
if not does_exist:
    try:
        os.makedirs(out_path)
    except:
        pass #TODO: why pass?


def load_lwe(n,q,eta,k,seed=0):
    #print(f"- - - k={k} - - - load")
    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_


def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads,dump_bkz=True):
    report = {
        "params": (n,q,eta,k,seed),
        "beta_bkz": beta_bkz,
        "sieve_dim": sieve_dim_max,
        "kappa": kappa,
        "bkz_runtime": 0,
        "bdgl_runtime": [0]*nsieves,
    }
    dim = n*k
    A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    if sieve_dim_max<60:
        nthreads = 1
    elif sieve_dim_max<80:
        nthreads = 2


    H11 = B[:len(B)-kappa] #the part of basis to be reduced
    H11 = IntegerMatrix.from_matrix( [ h11[:len(B)-kappa] for h11 in H11  ] )
    H11r, H11c = H11.nrows, H11.ncols
    assert(H11r==H11c)
    #for i in range(H11r):
    #    print(H11[i])
    #assert(False)

    LR = LatticeReduction( H11, threads_bkz=nthreads )
    bkz_start = time.perf_counter()
    for beta in range(5,beta_bkz+1):
        then_round=time.perf_counter()
        LR.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
    report["bkz_runtime"] = time.perf_counter() - bkz_start

    if dump_bkz:
        with open(out_path+f"/bkzdump_{n}_{q}_{eta}_{k}_{seed}_{kappa}_{sieve_dim_max-nsieves+i}", "wb") as f:
            pickle.dump({"B": H11}, f)


    #---------run sieving------------
    int_type = H11.int_type
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    G = GSO.Mat( H11, U=IntegerMatrix.identity(H11r,int_type=int_type), UinvT=IntegerMatrix.identity(H11r,int_type=int_type), float_type=ft )
    G.update_gso()
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(H11r-sieve_dim_max, H11r-sieve_dim_max+nsieves ,H11r)
    for i in range(nsieves):
        sieve_start = time.perf_counter()
        g6k(alg="bdgl")
        report["bdgl_runtime"][i] = time.perf_counter()-sieve_start
        print(f"siever-{kappa}-{sieve_dim_max-nsieves+i} finished in added time {time.perf_counter()-sieve_start}" )
        g6k.dump_on_disk(out_path+f"g6kdump_{n}_{q}_{eta}_{k}_{seed}_{kappa}_{sieve_dim_max-nsieves+i}")
        g6k.extend_left(1)

    return report

if __name__=="__main__":
    # (dimension, predicted kappa, predicted beta)
    params = [(140, 12, 48), (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    #params = [(140, 12, 48)]#, (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    nworkers, nthreads = 20, 4

    lats_per_dim = 10
    inst_per_lat = 10 #how many instances per A, q
    q, eta = 3329, 3
    #def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)
    output = []
    pool = Pool(processes = nworkers )
    tasks = []
    for param in params:
        for latnum in range(lats_per_dim):
            for instance in range(inst_per_lat):
                for kappa in range(param[1]-1, param[1]+2,1):
                    tasks.append( pool.apply_async(
                        run_preprocessing, (
                            param[0], #n
                            q, #q
                            eta, #eta
                            1, #k
                            [latnum,instance], #seed
                            param[2]-5, #beta_bkz
                            param[2]+2, #sieve_dim_max
                            7,  #nsieves
                            kappa, #kappa
                            nthreads #nthreads
                        )
                    ) )

    for t in tasks:
        output.append( t.get() )
    pool.close()

    for o_ in output:
        print(o_)
