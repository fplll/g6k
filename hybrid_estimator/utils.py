from math import sqrt, log

#Central Binomial probablity mass functions
CB2 = [(1/2)**5, 5*(1/2)**5, 10*(1/2)**5, 10*(1/2)**5, 5*(1/2)**5, (1/2)**5 ]
CB3 = [(1/2)**7, 7*(1/2)**7, 21*(1/2)**7, 35*(1/2)**7, 35*(1/2)**7, 21*(1/2)**7, 7*(1/2)**7, (1/2)**7]

def st_dev_central_binomial(eta):
    return sqrt(eta / 2.0)

def plot_gso(r, *args, **kwds):
    return line([(i, r_,) for i, r_ in enumerate(r)], *args, **kwds)

#entropy
def H(pr):
    return -sum([p*log(p,2) for p in pr])
