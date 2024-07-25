from zgsa_nonsym import ZGSA, ZGSA_old
from batchCVP import batchCVPP_cost
from parser import parse_all
from utils import st_dev_central_binomial, H, CB2, CB3

def plot_gso(r, *args, **kwds):
    return line([(i, r_,) for i, r_ in enumerate(r)], *args, **kwds)

#Thm. 4.1
def find_beta(d, n, q, st_dev_e):
    minbeta = 50 if d<513 else n//2
    for beta in range(minbeta, d//2, 1): #90, 450, 1
        r_log = ZGSA(d, n, q, beta)
        #r_log = ZGSA_old(d, n, q, beta)
        # plot_gso(r_log).show()
        if beta%32==0:
            plot_gso(r_log).save(f"bkz{beta}.png")
        lhs  = 0.5*log(beta)+log(st_dev_e)
        rhs  = r_log[2*n-beta] #counting from 0
        if lhs < rhs:
            return beta
        #print(beta, lhs.n(), rhs.n())
    return infinity

#core-SVP
def svp_cost(beta, d, alg="BDGL16_real"):
    if alg == "BDGL16_real":
        return 0.387*beta+log(8*d,2)+16.4
    elif alg == "BDGL16_asym":
        return 0.292*beta+log(8*d,2)+16.4
    else:
        print("Unrecognized SVP algorithm")
        return 0

if __name__=="__main__":
    n, q, kappa, st_dev_e, dist = parse_all()
    dim = 2*n


    kappa = 45  #max number of guessed coordiantes
    min_rt = infinity
    minTbkz = 0
    minTcvp = 0
    minbeta = 0
    minkappa = 0
    for kappa_ in range(15,kappa+1):
        M_log = kappa_*H(CB3)+1 # number of CVP-targets
        beta = find_beta(dim-kappa_, n-kappa_, q, st_dev_e)
        if beta==infinity: continue
        Tbkz = svp_cost(beta,dim-kappa_)
        Tcvp = batchCVPP_cost(beta, M_log, sqrt(4/3.), 1)
        min_ = max(Tbkz, Tcvp)
        if min_<min_rt:
            min_rt = min_
            minTbkz = Tbkz
            minTcvp = Tcvp
            minbeta = beta
            minkappa = kappa_
            print(RR(min_rt), RR(minTbkz), RR(minTcvp), minbeta, minkappa)

    print()
    print(f"n={n}, q={q}")
    print(f"Est. cost: {RR(min_rt):.4f}, Cost SVP: {RR(minTbkz):.4f}, Cost CVP: {RR(minTcvp):.4f}, beta: {minbeta}, guessing coords.: {minkappa}")
