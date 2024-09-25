#example  sage hybrid.sage -n "140:150:10" -q 3329 -eta 3

load("../framework/instance_gen.sage")
load("../batchCVP.py")


import argparse


def entropy(D):
    H = sum( [ -p*log(p,2) for p in D.values() ] )
    return H

parser = argparse.ArgumentParser()
parser.add_argument("n", type=str)
parser.add_argument("q", type=int)
parser.add_argument("eta", type=int)

args, unknown = parser.parse_known_args()
if len(unknown) > 0:
    print("Unknown arguments " + str(unknown) + " will be ignored.")

print(args.n)

nMin, nMax, nStep = args.n.split(":")
nMin = int(nMin)
nMax = int(nMax)
nStep = int(nStep)

q = args.q
eta = args.eta

D_s = build_centered_binomial_law(eta)
D_e = D_s

H = entropy(D_s)

costNonHybrid = 0
bestCosts = 0

print("n\tkappa\t\tBKZ only\t\tHybrid")

for n in range(nMin,nMax+1,nStep):
    for kappa in range(n):
        A, b, dbdd = initialize_from_LWE_instance(DBDD_predict_diag, n-kappa, q, n, D_e, D_s, verbosity = 0)
        dbdd.integrate_q_vectors(q)

        beta, delta = dbdd.estimate_attack()

        costBKZ = 0.292*beta + log(8*dbdd.dim(), 2) + 16.4
        _, costCVP = batchCVPP_cost(beta, H*kappa, sqrt(4/3), 1)

        if kappa == 0:
            costNonHybrid = costBKZ
            betaNonHybrid = beta

        if costCVP > costBKZ:
            break

        costHybrid = costBKZ+1
        betaHybrid = beta

    print("%d\t%d\t\t%d (%f)\t\t%d (%f)" % (n,kappa-1,betaNonHybrid,costNonHybrid, betaHybrid, costHybrid))
