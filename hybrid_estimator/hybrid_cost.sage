from math import lgamma
from zgsa_nonsym import ZGSA, ZGSA_old
from batchCVP import batchCVPP_cost



def st_dev_central_binomial(eta):
    return sqrt(eta / 2.0)

def plot_gso(r, *args, **kwds):
    return line([(i, r_,) for i, r_ in enumerate(r)], *args, **kwds)

#entropy
def H(pr):
    return -sum([p*log(p,2) for p in pr])

#Thm. 4.1
def find_beta(d, n, q, st_dev_e):
    for beta in range(300, 450, 1):
        r_log = ZGSA(d, n, q, beta)
        #r_log = ZGSA_old(d, n, q, beta)
        #plot_gso(r_log).show()
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

#Central Binomial probablity mass functions
CB2 = [(1/2)**5, 5*(1/2)**5, 10*(1/2)**5, 10*(1/2)**5, 5*(1/2)**5, (1/2)**5 ]
CB3 = [(1/2)**7, 7*(1/2)**7, 21*(1/2)**7, 35*(1/2)**7, 35*(1/2)**7, 21*(1/2)**7, 7*(1/2)**7, (1/2)**7]

Kyber512 = {'n': 2*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3}
Kyber768 = {'n': 3*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}
Kyber1024 = {'n': 4*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}


n = Kyber512['n']
q = Kyber512['q']
st_dev_e = Kyber512['st_dev_e']
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
        print(min_rt.n(), minTbkz.n(), minTcvp, minbeta, minkappa)
        
print(min_rt.n(), minTbkz.n(), minTcvp, minbeta,minkappa)
