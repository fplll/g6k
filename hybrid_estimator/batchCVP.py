from math import log, ceil, sqrt, exp

def batchCVPP_cost(d, M, alpha, gamma):
    # as per https://wesselvanwoerden.com/publication/ducas-2020-randomized/randomized_slicer.pdf
    #https://eprint.iacr.org/2016/888.pdf gives the same numbers since we have sqrt(4/3) short vectors given
    #alpha = sqrt(4/3.) #1.00995049383621 #<2*c - 2*sqrt(c^2-c) < 1.10468547559928 since a <1.22033 (see p.22)
    #gamma = 1     #gamma-CVP
    a = alpha**2
    b = a**2/(4*a - 4)
    c = gamma**2
    assert(b>c)
    n = ceil(-1/2 + sqrt((4*b-a)**2-8*c*(2*b-a))/(2*a))

    #print(a, b, c, n)

    #Eq.(12)
    def p(a, x, y):
        return sqrt(a - (a+x-y)**2/(4*x))

    #Eq.(16)
    def omega(a, x, y):
        return -log(p(a,x,y))

    if n==1:
        u = 0
        v = c-b

    else:
        disc = (a*n**2 - (b+c))**2 + 4*b*c*(n**2-1)
        u = ( (b+c-a)*n - sqrt(disc) )/(n**3 - n)
        v = ( (a-2*b)*n**2 + (b-c) + n*sqrt(disc) )/(n**3-n)


    # Eq.(43) success probablity for one target
    prob = 0
    x_ = [0]*(n+1)
    x_[0] = b
    x_[n] = c
    for i in range(1, n+1):
        x_[i] = u*i**2+v*i+b
        prob += omega(a, x_[i-1], x_[i])
        #print(i, x_[i].n(), prob)

    T = (a - 2*(a-1)/(1+sqrt(1-1./a)))**(-1/2.)  #base for power-d

    prob = exp(-prob) #1/prob=number of rerandomizations per target

    N_batches = ceil(M - 1./prob) #M and prob are on log2-scale

    T_per_batch_log = d*log(1./prob*T,2)
    T = d*log(1./prob*T,2) + N_batches

    #print("prob:", prob)  # 0.901387818865997 for a = 4/3
    #print("T:", T)        # 1.06066017177982 for a = 4/3
    #print("RT:", 1./prob*T, log(1./prob*T, 2).n()) #1.17669681082910 for a = 4/3
    return T
