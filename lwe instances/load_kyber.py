import sys,os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
path = "saved_lattices/"

# def se_gen(k,n,eta):
#     s = binomial_vec(k*n, eta)
#     e = binomial_vec(k*n, eta)
#     return s, e
#
# def generateLWEInstances(n, q = 3329, eta = 3, k=1, ntar=5):
#     A,q = kyberGen(n,q = q, eta = eta, k=k)
#     bse = []
#     for _ in range(ntar):
#         s, e = se_gen(k,n,eta)
#         b = (s.dot(A) + e) % q
#         bse.append( (b,s,e) )
#
#     return A,q,bse
#
# def gen_and_dump_lwe(n, q, eta, k, ntar, seed=0):
#     A,q,bse= generateLWEInstances(n, q, eta, k, ntar)
#
#     with open(f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "wb") as fl:
#         pickle.dump({"A": A, "q": q, "eta": eta, "k": k, "bse": bse}, fl)

def load_lwe(n,q,eta,k,seed=0):
    print(f"- - - k={k} - - - load")
    with open(path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_

def load_and_dump_kyber(n,q,eta,k,betamax,seed=[0,0]):
    # prepeare the lattice
    print( f"loading {n,q,eta,k}" )
    # try:
    A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    # b, s, e = bse[seed[1]]

    r,c = A.shape
    print(f"Shape: {A.shape}, n, k: {n,k}")

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ): #qary part
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n): #copy A
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    dim = len(B)
    print(f"dim: {dim}")
    B = IntegerMatrix.from_matrix(B)
    # make GSO
    FPLLL.set_precision(250)
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    with open(path + f"kyber_lat_{n}_{q}_{eta}_{k}_{seed[0]}", "wb") as file:
        pickle.dump({"B": B, "eta": eta, "k": k, "bse": bse}, file)

    # G = GSO.Mat(
    #     LR.gso.B, U=IntegerMatrix.identity(dim-kappa,int_type=int_type),
    #     UinvT=IntegerMatrix.identity(dim-kappa,int_type=int_type), float_type=ft
    # )


isExist = os.path.exists(path)
if not isExist:
    try:
        os.makedirs(path)
    except:
        pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

nthreads = 40
lats_per_dim = 10 #10
inst_per_lat = 10 #10 #how many instances per A, q
q, eta = 3329, 3

nks = [ (140+10*i,3) for i in range(3) ]
output = []
pool = Pool(processes = nthreads )
tasks = []

for nk in nks:
    n, k = nk[0], 1
    for latnum in range(lats_per_dim):
        # gen_and_dump_lwe(nk[0], q, eta,k, nk[1], latnum)
        for tstnum in range(inst_per_lat):
            tasks.append( pool.apply_async(
                load_and_dump_kyber, (nk[0],q,eta,k,80,[latnum,tstnum])
                ) )

for t in tasks:
        output.append( t.get() )

pool.close()
