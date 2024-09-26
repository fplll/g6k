from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys
from time import perf_counter
from experiments.lwe_gen import *

from hyb_att_on_kyber import alg_3, alg_2_batched
from sample import *

inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"

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

def alg_3_ans(g6k,B,H11,t,n_guess_coord, eta, ans, nthreads=1, tracer_alg3=None):
    # raise NotImplementedError
    # - - - prepare targets - - -
    then_start = perf_counter()
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = t[:dim-n_guess_coord], t[dim-n_guess_coord:]
    slicer = RandomizedSlicer(g6k)
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    nsampl = 1 #min(max_nsampl, nsampl)
    target_candidates = [t1] #first target is always the original one
    vtilde2s = [np.array((dim-n_guess_coord)*[0] + list(t2))]

    # B[:dim-n_guess_coord][0][:dim-n_guess_coord] #this does not work
    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    for times in range(nsampl): #Alg 3 steps 4-7
        if times!=0 and times%64 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        etilde2 = ans[-n_guess_coord:] #np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        # print(f"len etilde2: {len(etilde2)}")
        vtilde2 = np.array(t2)-etilde2
        tmp = np.concatenate([(dim-n_guess_coord)*[0] , vtilde2])
        vtilde2s.append( tmp )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = H12.multiply_left(vtilde2)
        # tmp = t - np.concatenate( [tmp, n_guess_coord*[0]] )

        # print(f"len(vtilde2): {len(vtilde2)} len(t1): {len(t1)}")
        # print(f"dim: {dim} n_guess_coord: {n_guess_coord}")
        t1_ = np.array( list(t1) ) - tmp
        # print(t1_)
        # print(f"len t1_: {len(t1_)}")
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: dist_sq_bnd might have changed at this point (or even in attacker)
    #TODO: deduce what is the betamax
    betamax = 48
    ctilde1 = alg_2_batched( g6k,target_candidates, nthreads=nthreads, tracer_alg2=None )

    assert len(ctilde1) == H11.nrows
    v1 = np.array( H11.multiply_left( ctilde1 ) )
    #keep a track of v2?
    argminv = None
    minv = 10**12
    for vtilde2 in vtilde2s:
        v2 = vtilde2
        v = np.concatenate([v1,n_guess_coord*[0]])+v2
        vv = v@v
        # print(f"v: {v}")
        if vv < minv:
            argminv = v
    return v

if __name__=="__main__":
    n, k = 70, 1
    eta = 3
    n_guess_coord, n_slicer_coord = 5, 50
    sieve_dim_max = n_slicer_coord
    nsieves = 2
    nthreads = 2
    betamax = 40
    A,q,bse = generateLWEInstances(n, q = 3329, eta = eta, k=k, ntar=1)
    b, s, e = bse[0]

    Binit = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        Binit[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        Binit[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            Binit[i][j] = int( A[i-k*n,j] )

    dim = 2*k*n
    ft = "ld" if k*n<99 else ( "dd" if config.have_qd else "mpfr")
    H11 = IntegerMatrix.from_matrix( [b[:dim - n_guess_coord] for b in Binit[:dim - n_guess_coord] ] )

    LR = LatticeReduction( H11, nthreads )
    for beta in range(5,betamax+1):
        then = perf_counter()
        LR.BKZ( beta )
        print(f"BKZ-{beta} done in {perf_counter()-then}")

    H11r, H11c = H11.nrows, H11.ncols
    G = GSO.Mat( H11,U=IntegerMatrix.identity(H11r), UinvT=IntegerMatrix.identity(H11r), float_type=ft )
    G.update_gso()
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(H11r-sieve_dim_max, H11r-sieve_dim_max+nsieves ,H11r)

    t = np.concatenate([b,n*[0]])
    answer = np.concatenate( [(s.dot(A)) % q,(dim//2)*[0]] )
    g6k(alg="bdgl")

    for i in range(n-n_guess_coord):
        for j in range(n-n_guess_coord):
            Binit[i][j] = int( H11[i][j] ) #bkz reduce B up to n-n_guess_coord
    B = IntegerMatrix.from_matrix(Binit)
    # print(B)

    v = alg_3(g6k,B,H11,t,n_guess_coord, eta, nthreads=nthreads, tracer_alg3=None)
    # v = alg_3_ans(g6k,B,H11,t,n_guess_coord, eta, ans=np.concatenate([-e,s]), nthreads=nthreads, tracer_alg3=None)
    print(answer)
    print(v)
    # print(([-s,e])) #np.concatenate
