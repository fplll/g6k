import numpy as np
import random
from numpy import array, zeros, copy
from math import *
import sys
from time import time

Tlim = 1000
Sample = 1000


n = int(sys.argv[1])

relative_len = False
try:
    XPC_bitlen = int(sys.argv[2])
    XPC_bitlen_string = str(XPC_bitlen)
except:
    XPC_bitlen_rel = float(sys.argv[2])
    XPC_bitlen_string = str(XPC_bitlen_rel)+"n"
    XPC_bitlen = int(ceil(2**(XPC_bitlen_rel * n)))

R = sqrt(4./3)


def rand_unif_sphere(r=1.0):
    v = array([random.gauss(0, 1.) for i in range(n)], dtype=np.float64)
    l = v.dot(v)
    return (r / sqrt(l))*v


def random_pair():
    return (rand_unif_sphere(R), rand_unif_sphere(R))


def good_pair():
    v = rand_unif_sphere(R)
    w = rand_unif_sphere()
    w -= (w.dot(v)/R**2) * v
    w *= (1. / sqrt(w.dot(w)))
    w += .5 * v
    return v, w


def random_sparse_code(c, s=6):
    M = zeros((c, n), dtype=np.int64)
    for v in M:
        for rep in xrange(s):
            while True:
                a = random.randint(0, n-1)
                if v[a]==0:
                    v[a] = 2 * (rep % 2) - 1
                    break
    return M

def compress(M, u):
    return M.dot(u) > 0

def mesure_XPC_quality(M, S=Sample):
    l = len(M)
    hist_good = zeros(l+1, dtype=np.int64)
    hist_bad = zeros(l+1, dtype=np.int64)
    for a in xrange(S):
        v, w = random_pair()
        x = sum(compress(M, v) ^ compress(M, w))
        hist_bad[x] += 1

        v, w = good_pair()
        x = sum(compress(M, v) ^ compress(M, w))
        hist_good[x] += 1

    for i in range(l):
        hist_good[i+1] += hist_good[i]
        hist_bad[i+1] += hist_bad[i]

    return hist_bad, hist_good


def line_scores(G):
    s = G[0, 0] * G[0, 0]
    n, _ = G.shape
    v = [(sum([x*x for x in G[i]]) - s,i) for i in range(n)]
    return v


def score(G):
    s = G[0, 0] * G[0, 0]
    v = [sum([x*x for x in v]) - s for v in G]
    return sum([x*x for x in v])


def update_G(M, G, i):
    l, n = M.shape
    v = M.dot(M[i].transpose())
    for j in xrange(l):
        G[i, j] = v[j]
        G[j, i] = v[j]


def perm(n):
    L = range(n)
    for rep in xrange(2 * n):
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        L[a], L[b] = L[b], L[a]
    return L


def improve_once(M, G, i):
    l, n = M.shape
    old_score = score(G)
    pp = perm(n)
    for aa in range(n):
        a = pp[aa]
        for bb in range(a):
            b = pp[bb]
            if M[i, a] == M[i, b]:
                continue
            (M[i, a], M[i, b]) = (M[i, b], M[i, a])
            update_G(M, G, i)
            new_score = score(G)
            if new_score < old_score:
                return True
            else:
                (M[i, a], M[i, b]) = (M[i, b], M[i, a])
    update_G(M, G, i)
    return False


def improve(M):
    l, n = M.shape
    G = M.dot(M.transpose())
    score0 = score(G)
    T0 = time()
    a = 0
    while True:
        a += 1
        if time() - T0 > Tlim:
            break 
        v = line_scores(G)
        v.sort(reverse=True)
        c = 0
        for (_, i) in v:
            if improve_once(M, G, i):
                break
            else:
                c += 1
        else:
            print "no improvement found"
            break
    G = M.dot(M.transpose())
    print "n:", n , "iter:", a, " \t" , 1.*score0, "\t ->", 1.*score(G)
    return M


def get_good_code(s=6):
    M = random_sparse_code(XPC_bitlen, s=s)
    # print mesure_XPC_quality(M)
    improve(M)
    return M

M = get_good_code(s=6)

def sparse_repr(v):
    Lp = []
    Lm = []
    for i in range(n):
        if v[i] > 0:
            Lp += [i]
        if v[i] < 0:
            Lm += [i]
    return Lp + Lm

f = open("sc_%d_%s.def"%(n, XPC_bitlen_string), 'w')

for v in M:
    for x in sparse_repr(v):
        print >>f, x,
    print >>f

f.close()


# f = open("sc_%d_%s.bench"%(n, XPC_bitlen_string), 'w')

# b, g = mesure_XPC_quality(M)
# for i in range(XPC_bitlen):
#     print >>f, "%d \t %d \t %d"%(i, b[i], g[i])

# f.close()
