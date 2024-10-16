from experiments.lwe_gen import *

import sys,os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from sample import *

from preprocessing import run_preprocessing
#def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)

approx_fact = 1.07

max_nsampl = 150 #10**7
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

def gen_and_dump_lwe(n, q, eta, k, ntar, seed=0):
    A,q,bse= generateLWEInstances(n, q, eta, k, ntar)

    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "wb") as fl:
        pickle.dump({"A": A, "q": q, "eta": eta, "k": k, "bse": bse}, fl)

def load_lwe(n,q,eta,k,seed=0):
    print(f"- - - k={k} - - - load")
    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_

def prepare_kyber(n,q,eta,k,betamax,kappa,seed=[0,0]): #for debug purposes
    report = {
        "kyb": ( n,q,eta,k ),
        "beta": 0,
        "kappa": 0,
        "time": 0
    }
    # prepeare the lattice
    dim = n*k
    print( f"Launching hybrid on: {n,q,eta,k}" )
    print(f"betamax,kappa: {betamax,kappa}")
    try:
        A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
        filename = f"g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+i}"
        g6k = Siever.restore_from_file( filename )
    except FileNotFoundError:
        gen_and_dump_lwe(n, q, eta, k, ntar, seed[0])
        A, q, eta,k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    print(f"lenbse: {len(bse)} seed={seed[1]}")
    b, s, e = bse[seed[1]]

    r,c = A.shape
    print(f"Shape: {A.shape}, n, k: {n,k}")
    t = np.concatenate([b,[0]*r]) #BDD target
    x = np.concatenate([b-e,s]) #BBD solution
    sol = np.concatenate([e,-s])

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    tarnrmsq = 1.01*(sol.dot(sol))

    H11 = deepcopy( B[B.nrows-kappa] ) #the part of basis to be reduced
    LR = LatticeReduction( H11 )
    for beta in range(5,betamax+1):
        then_round=time.perf_counter()
        LR.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
    return { 'B': B, 'H11': H11, 'q': q, 'eta': eta, 'k': k, 'bse': bse, 'betamax': betamax }


def attacker(input_dict, n_guess_coord, sieve_dim_max, nsieves, nthreads=1, tracer_exp=None):
    """
    # param input_dict: dictionary
    param n_guess_coord: guessing stage dim (kappa)
    param dist_sq_bnd: distance bound
    param sieve_dim_max: 1 + max dim siever is prepared for
    param nsieves: length of range within which we attempt to find a solution
    Assumes H11 is BKZ-betamax reduced while B is original. Assumes all Sievers are precomputed
    and stored in out_path+f"/bkzdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+i}"
    for i in range(nsieves).
    """
    # B, H11, q, eta, k, bse, betamax = input_dict['B'], input_dict['H11'], input_dict['q'], input_dict['eta'], input_dict['k'], input_dict['bse'], input_dict['betamax']
    n, kappa, q, eta, k, seed = input_dict['n'], n_guess_coord, input_dict['q'], input_dict['eta'], input_dict['k'], input_dict['seed']
    A, q, eta, k, bse = load_lwe(n,q,eta,k,seed=seed[0])

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    g6k = Siever.restore_from_file( out_path + f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-1}' )
    H11 = g6k.M.B
    B = IntegerMatrix.from_matrix(B)

    dim = 2*k*n
    fl = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

    int_type = B.int_type
    ft = "dd" if config.have_qd else "mpfr"

    for sieveid in range(nsieves):
        vec_index = 0
        filename_siever = out_path+f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+sieveid}'
        g6k = Siever.restore_from_file(filename_siever)
        # g6k.params["nthreads"] = nthreads #readonly
        for b, s, e in bse:
            try:
                candidate = alg_3(g6k,B,H11,np.concatenate([b,n*[0]]),n_guess_coord, eta, nthreads=nthreads, tracer_alg3=None)
                answer = np.concatenate( [(s.dot(A)) % q,(dim//2)*[0]] )
                print( f"answer: {answer}" )
                print( f"candidate: {candidate}" )
                #TODO: automate the check
                if all(answer==candidate):
                    print(f"vec-{vec_index}: succsess!")
                else:
                    print(print(f"vec-{vec_index}: fail!"))
                vec_index += 1
            except NotImplementedError as expt:
                print("g6k load supposedly successfull.")
                pass

def alg_3(g6k,B,H11,t,n_guess_coord, eta, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    #TODO: inject correct target and see what happens
    # raise NotImplementedError
    # - - - prepare targets - - -
    then_start = perf_counter()
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = t[:-n_guess_coord], t[-n_guess_coord:]
    slicer = RandomizedSlicer(g6k)
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = [t1] #first target is always the original one
    vtilde2s = [np.array(t2) ]

    # B[:dim-n_guess_coord][0][:dim-n_guess_coord] #this does not work
    # H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    # for times in range(nsampl): #Alg 3 steps 4-7
    #     if times!=0 and times%64 == 0:
    #         print(f"{times} done out of {nsampl}", end=", ")
    #     etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
    #     # print(f"len etilde2: {len(etilde2)}")
    #     vtilde2 = np.array(t2)-etilde2
    #     tmp = np.concatenate([(dim-n_guess_coord)*[0] , vtilde2])
    #     vtilde2s.append( vtilde2   )
    #     #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
    #     tmp = H12.multiply_left(vtilde2)
    #
    #     # print(f"len(vtilde2): {len(vtilde2)} len(t1): {len(t1)}")
    #     # print(f"dim: {dim} n_guess_coord: {n_guess_coord}")
    #     t1_ = t1 - tmp #
    #     # print(t1_)
    #     # print(f"len t1_: {len(t1_)}")
    #     target_candidates.append( t1_ )
    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    for times in range(nsampl): #Alg 3 steps 4-7
        if times!=0 and times%64 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        # print(f"len etilde2: {len(etilde2)}")
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = H12.multiply_left(vtilde2)

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
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

    v1 = np.array( g6k.M.B[:len(ctilde1)].multiply_left( ctilde1 ) )
    #keep a track of v2?
    argminv = None
    minv = 10**12
    cntr=0
    for vtilde2 in vtilde2s:
        # t = target_candidates[cntr]
        tmp = H12.multiply_left(vtilde2)
        v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
        v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
        v_t = v - np.array(t) #-np.concatenate([t,n_guess_coord*[0]])
        vv = v_t@v_t
        # print(f"v: {v}")
        if vv < minv:
            minv = vv
            argminv = v
        cntr += 1
    return argminv

def alg_2_batched( g6k,target_candidates, dist_sq_bnd=1.0, nthreads=1, tracer_alg2=None ):
    # raise NotImplementedError
    sieve_dim = g6k.r-g6k.l #n_slicer_coord
    print(f"in alg2 sieve_dim={sieve_dim}")

    # dist_sq_bnd = 1.0 #TODO: implement
    G = g6k.M
    B = G.B
    dim = G.d
    Gsub = GSO.Mat( G.B[:dim-sieve_dim], float_type=G.float_type )
    Gsub.update_gso()
    # print(f"dim(Gsub): {Gsub.d}")

    # - - - prepare Slicer for batch cvp - - -
    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(1);
    # - - - END prepare Slicer for batch cvp - - -
    scaling_vec = np.array( [tmp**0.5 for tmp in G.r()[dim-sieve_dim:]] )
    #WARNING: we do not store t_gs_reduced_list since t_gs_list =  t_gs - gs(shift_babai_c*B)
    #this is a time-memory tradeoff. Since Slicer returns only an error vector, we don\'t
    #know which of the target candidates it corresponds to. TODO: or should we?
    target_list_size =  2 * g6k.db_size() #len(g6k)
    nrand = 150 #min( 250, target_list_size / len(target_candidates ) )
    print(f"len(target_candidates): {len(target_candidates)} nrand: {nrand}")
    t_gs_list = []
    t_gs_reduced_list = []
    shift_babai_c_list = []
    for target in target_candidates:
        # print(end=".", flush=True)
        t_gs = from_canonical_scaled( G,target,offset=sieve_dim )
        t_gs_non_scaled = G.from_canonical(target)[dim-sieve_dim:]
        shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, dimension=sieve_dim, gso=True) )
        print( f"shift_babai_c: {shift_babai_c}" )
        shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
        t_gs_reduced = from_canonical_scaled( G,np.array(target)-shift_babai,offset=sieve_dim ) #this is the actual reduced target
        assert len(t_gs_reduced) == sieve_dim
        assert all( abs( t_gs_reduced[dim-sieve_dim:] ) <0.501 ) #assert that the last Sieve dim coords are size reduced

        t_gs_list.append(t_gs)
        shift_babai_c_list.append(shift_babai_c)
        t_gs_reduced_list.append(t_gs_reduced)

        # print(target[dim-sieve_dim:])
        # print(f"Doing grow_db")
        then_gdbwt = perf_counter()
        print(f"supposed ce.len: {t_gs_reduced@t_gs_reduced}")
        slicer.grow_db_with_target(t_gs_reduced, n_per_target=nrand)
        # slicer.grow_db_with_target((dim-sieve_dim)*[0] + [float(tt) for tt in t_gs_reduced[dim-sieve_dim:]], n_per_target=nrand) #add a candidate to the Slicer
        gdbwt_t = perf_counter() - then_gdbwt #TODO: collect this stat
        # print(f"grow_db done in {gdbwt_t}",flush=True)
    #run slicer
    print(f"running slicer")
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))
    # slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (approx_fact*approx_fact*(dist_sq_bnd)))
    print(f"t_gs_reduced: {t_gs_reduced}")
    iterator = slicer.itervalues_t()
    for tmp in iterator:
        out_gs_reduced = np.array(tmp)  #db_t[0] is expected to contain the error vector
        cur_nrm_sq = out_gs_reduced@out_gs_reduced
        break

    print(f"out_gs_reduced-t_gs_reduced: {out_gs_reduced-t_gs_reduced}")
    print(f"out_gs_reduced: {out_gs_reduced}")
    index = 0
    #Now we deduce which target candidate the error vector corresponds to.
    #The idea is that if t_gs is an answer then t_gs_reduced - out_gs_reduced is in the projective lat
    #and is (close to) zero.
    min_norm_err_sq = float("inf")
    index_best = None
    b_best = None
    for index in range(len(shift_babai_c_list)):
        """
        t_gs_reduced = t_gs_reduced_list[index] #we could do this to t_gs, but this one is shorter
        b_gs_reduced = from_canonical_scaled( G,t_gs_reduced-out_gs_reduced,offset=sieve_dim )
        b_gs_reduced = b_gs_reduced / scaling_vec
        b_reduced = G.babai(list(b_gs_reduced), start=dim-sieve_dim, dimension=sieve_dim, gso=True)
        b_reduced = G.B.multiply_left( (dim-sieve_dim)*[0] + list( b_reduced ) )

        shift_babai_c_reduced =  shift_babai_c_list[index]
        shift_babai_reduced = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c_reduced ) )

        b = np.array( shift_babai_reduced ) + np.array( b_reduced )
        t = target_candidates[index]

        diff = np.array( t ) - np.array( b ) #an actual error vector we observe == actual error (+ some lattice vector for bad candidates)
        diff_gs_nrm_sq = diff@diff #its norm. Ideally, == norm of error
        """
        t_gs_reduced = t_gs_reduced_list[index] #we could do this to t_gs, but this one is shorter
        shift_babai_c_reduced =  shift_babai_c_list[index]

        shift_babai_reduced = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c_reduced ) )
        shift_babai_reduced_gs = from_canonical_scaled( G,shift_babai_reduced, offset=sieve_dim )
        guess = np.array(t_gs_reduced - out_gs_reduced)
        print(len(guess),len(shift_babai_reduced_gs))
        guess = guess + shift_babai_reduced_gs

        t_gs = t_gs_list[index]
        diff_gs = t_gs - guess #an actual error vector we observe == actual error (+ some lattice vector for bad candidates)
        diff_gs_nrm_sq = diff_gs@diff_gs #its norm. Ideally, == norm of error

        if diff_gs_nrm_sq < min_norm_err_sq:
            min_norm_err_sq = diff_gs_nrm_sq
            index_best = index
            # b_best = b

    print(f"min_norm_err_sq: {min_norm_err_sq}, index_best:{index_best}")
    index = index_best
    t = np.array( target_candidates[index] )
    #we substitute the obtaied error from the target and call babai to
    #account for an fp error
    # t_new = t - to_canonical_scaled( G, np.concatenate( [(dim-sieve_dim)*[0], out_gs_reduced] ) )
    t_new = t - to_canonical_scaled( G, out_gs_reduced, offset=sieve_dim )
    assert len(t_new) == dim
    bab_01 = G.babai(t_new)
    # bab_01 = G.babai(b_best)

    print(f"alg2 terminates")
    return bab_01


if __name__=="__main__":
    # (dimension, predicted kappa, predicted beta)
    # params = [(140, 12, 48), (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    # params = [(140, 12, 48)]#, (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    params = [(80, 2, 40)]
    nworkers, nthreads = 1, 1

    lats_per_dim = 2 #1
    inst_per_lat = 2 #10 #how many instances per A, q
    q, eta = 3329, 3
    #def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)
    output = []
    pool = Pool(processes = nworkers )
    tasks = []
    for param in params:
        for latnum in range(lats_per_dim):
            input_dict = {"n": param[0], "q":q , "eta":eta , "k":1 , "seed":[latnum,0] }
            tasks.append( pool.apply_async(
                attacker, (
                    input_dict, #input_dict
                    param[1], #n_guess_coord
                    param[2]+4, #sieve_dim_max
                    2, #7,  #nsieves
                    nthreads, #nthreads
                    None, #tracer_exp
                    )
            ) )

    for t in tasks:
        output.append( t.get() )
    pool.close()

    for o_ in output:
        print(o_)
    sys.stdout.flush()
