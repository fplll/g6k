from math import lgamma, log, exp, pi, floor
from time import perf_counter

# Low beta Gaussian Heuristic constant for use in NTRU Dense sublattice estimation.
gh_constant = {1: 0.00000, 2: -0.50511, 3: -0.46488, 4: -0.39100, 5: -0.29759, 6: -0.24880, 7: -0.21970, 8: -0.15748,
               9: -0.14673, 10: -0.07541, 11: -0.04870, 12: -0.01045, 13: 0.02298, 14: 0.04212, 15: 0.07014,
               16: 0.09205, 17: 0.12004, 18: 0.14988, 19: 0.17351, 20: 0.18659, 21: 0.20971, 22: 0.22728, 23: 0.24951,
               24: 0.26313, 25: 0.27662, 26: 0.29430, 27: 0.31399, 28: 0.32494, 29: 0.34796, 30: 0.36118, 31: 0.37531,
               32: 0.39056, 33: 0.39958, 34: 0.41473, 35: 0.42560, 36: 0.44222, 37: 0.45396, 38: 0.46275, 39: 0.47550,
               40: 0.48889, 41: 0.50009, 42: 0.51312, 43: 0.52463, 44: 0.52903, 45: 0.53930, 46: 0.55289, 47: 0.56343,
               48: 0.57204, 49: 0.58184, 50: 0.58852}

# Low beta \alpha_\beta quantity as defined in [AC:DucWoe21] for use in NTRU Dense subblattice estimation.
small_slope_t8 = {2: 0.04473, 3: 0.04472, 4: 0.04402, 5: 0.04407, 6: 0.04334, 7: 0.04326, 8: 0.04218, 9: 0.04237,
                  10: 0.04144, 11: 0.04054, 12: 0.03961, 13: 0.03862, 14: 0.03745, 15: 0.03673, 16: 0.03585,
                  17: 0.03477, 18: 0.03378, 19: 0.03298, 20: 0.03222, 21: 0.03155, 22: 0.03088, 23: 0.03029,
                  24: 0.02999, 25: 0.02954, 26: 0.02922, 27: 0.02891, 28: 0.02878, 29: 0.02850, 30: 0.02827,
                  31: 0.02801, 32: 0.02786, 33: 0.02761, 34: 0.02768, 35: 0.02744, 36: 0.02728, 37: 0.02713,
                  38: 0.02689, 39: 0.02678, 40: 0.02671, 41: 0.02647, 42: 0.02634, 43: 0.02614, 44: 0.02595,
                  45: 0.02583, 46: 0.02559, 47: 0.02534, 48: 0.02514, 49: 0.02506, 50: 0.02493, 51: 0.02475,
                  52: 0.02454, 53: 0.02441, 54: 0.02427, 55: 0.02407, 56: 0.02393, 57: 0.02371, 58: 0.02366,
                  59: 0.02341, 60: 0.02332}

# @cached_function
def ball_log_vol(n):
    return ((n/2.) * log(pi) - lgamma(n/2. + 1))

def log_gh(d, logvol=0):
    if d < 49:
        return (gh_constant[d] + logvol/d)

    return (1./d * (logvol - ball_log_vol(d)))

def delta(k):
    assert k >= 60
    delta = exp(log_gh(k)/(k-1))
    return (delta)

# @cached_function
def slope(beta):
    if beta<=60:
        return small_slope_t8[beta]
    if beta<=70:
        # interpolate between experimental and asymptotics
        ratio = (70-beta)/10.
        return ratio*small_slope_t8[60]+(1.-ratio)*2*log(delta(70))
    else:
        return 2 * log(delta(beta))

def ZGSA_old(d, n, q, beta, xi=1, tau=1, dual=False):
    from math import lgamma
    # from util import gh_constant, small_slope_t8
    """
    Reduced lattice Z-shape following the Geometric Series Assumption as specified in
    NTRU fatrigue [DucWoe21]
    :param d: Lattice dimension.
    :param n: The number of `q` vectors is `d-n`.
    :param q: Modulus `q`
    :param beta: Block size β.
    :param xi: Scaling factor ξ for identity part.
    :param dual: ignored, since GSA is self-dual: applying the GSA to the dual is equivalent to
           applying it to the primal.
    :returns: Logarithms (base e) of Gram-Schmidt norms
    """

    if not tau:
        L_log = (d - n)*[(log(q))] + n * [(log(xi))]
    else:
        L_log = (d - n)*[(log(q))] + n * [(log(xi))] + [(log(tau))]

    slope_ = slope(beta)
    diff = slope(beta)/2.

    for i in range(d-n):
        if diff > ((log(q)) - (log(xi)))/2.:
            break

        low = (d - n)-i-1
        high = (d - n) + i
        if low >= 0:
            L_log[low] = ((log(q)) + (log(xi)))/2. + diff

        if high < len(L_log):
            L_log[high] = ((log(q)) + (log(xi)))/2. - diff

        diff += slope_

    # Output basis profile as squared lengths, not ln(length)
    # L = [exp(2 * l_) for l_ in L_log]
    # return L

    # tmp = sum( rr for rr in L_log )
    # t0, t1 = tmp, log(q)*(d-n)
    # assert abs(t0-t1) < 10**-9, f"tst fail: {t0}, {t1.n()}"
    return L_log

def ZGSA( d,n,q,beta ):
    """
    param d: dimension of a lattice
    param n: such that there are d-n q-ary vectors
    param q: modulus
    param beta: blocksize
    returns log-profile (base e) of bkz-beta reduced basis
    """
    logq = log(q)
    l = (d-n)*[log(q)] + n*[0]
    slope_ = slope(beta)
    nGSA = floor( logq/((slope_)) )

    mindiff = 2**32
    logdet = (d-n)*logq
    jopt = -1
    for j in range(d):
        curdet = j*logq + nGSA*logq - nGSA*(nGSA-1)/2*slope_ # (nGSA*logq - nGSA*(nGSA-1)/2*slope_) = sum( (logq-i*slope_) for i in range(nGSA) )
        tmp = abs( logdet - curdet )
        if tmp < mindiff:
            mindiff = tmp
            jopt = j
        else:   #local min. is reached
            break

    sc = (logdet - (jopt*logq + nGSA*logq - nGSA*(nGSA-1)/2*slope_)) / nGSA  #scaling coeff
    l = jopt*[logq] + [ (logq-i*slope_)+sc for i in range(min(nGSA,d-jopt)) ] + (d-jopt-nGSA)*[0] #restore the profile

    return l

if __name__=="__main__":
  import matplotlib.pyplot as plt
  import numpy as np

  d, n = 1536, 900 #d-n qary
  q = 2048
  beta = 130

  then = perf_counter()
  lmy = ZGSA( d,n,q,beta )
  print(f"Sliding window done in {perf_counter()-then}")

  then = perf_counter()
  lold = ZGSA_old(d, n, q, beta, xi=1, tau=1)
  print(f"M.Alb. done in {perf_counter()-then}")

  plt.plot( [i for i in range(len(lmy))], lmy, color="blue")
  plt.savefig("prof_my.png")

  plt.plot([i for i in range(len(lold))],lold ,color="red")
  plt.savefig("prof_old.png")
