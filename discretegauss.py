# Implementation of exact discrete gaussian distribution sampler
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import random #Default random number generator,
#random.SecureRandom() provides high-quality randomness from /dev/urandom or similar
from fractions import Fraction #we will work with rational numbers

#sample uniformly from range(m)
#all randomness comes from calling this
def sample_uniform(m,rng):
    assert isinstance(m,int) #python 3
    #assert isinstance(m,(int,long)) #python 2
    assert m>0
    return rng.randrange(m)

#sample from a Bernoulli(p) distribution
#assumes p is a rational number in [0,1]
def sample_bernoulli(p,rng):
    assert isinstance(p,Fraction)
    assert 0 <= p <= 1
    m=sample_uniform(p.denominator,rng)
    if m < p.numerator:
        return 1
    else:
        return 0

#sample from a Bernoulli(exp(-x)) distribution
#assumes x is a rational number in [0,1]
def sample_bernoulli_exp1(x,rng):
    assert isinstance(x,Fraction)
    assert 0 <= x <= 1
    k=1
    while True:
        if sample_bernoulli(x/k,rng)==1:
            k=k+1
        else:
            break
    return k%2

#sample from a Bernoulli(exp(-x)) distribution
#assumes x is a rational number >=0
def sample_bernoulli_exp(x,rng):
    assert isinstance(x,Fraction)
    assert x >= 0
    #Sample floor(x) independent Bernoulli(exp(-1))
    #If all are 1, return Bernoulli(exp(-(x-floor(x))))
    while x>1:
        if sample_bernoulli_exp1(Fraction(1,1),rng)==1:
            x=x-1
        else:
            return 0
    return sample_bernoulli_exp1(x,rng)

#sample from a geometric(1-exp(-x)) distribution
#assumes x is a rational number >= 0
def sample_geometric_exp_slow(x,rng):
    assert isinstance(x,Fraction)
    assert x >= 0
    k=0
    while True:
        if sample_bernoulli_exp(x,rng)==1:
            k=k+1
        else:
            return k
            
#sample from a geometric(1-exp(-x)) distribution
#assumes x >= 0 rational
def sample_geometric_exp_fast(x,rng):
    assert isinstance(x,Fraction)
    if x==0: return 0 #degenerate case
    assert x>0

    t=x.denominator
    while True:
        u=sample_uniform(t,rng)
        b=sample_bernoulli_exp(Fraction(u,t),rng)
        if b==1:
            break
    v=sample_geometric_exp_slow(Fraction(1,1),rng)
    value = v*t+u
    return value//x.numerator
    
#sample from a discrete Laplace(scale) distribution
#Returns integer x with Pr[x] = exp(-abs(x)/scale)*(exp(1/scale)-1)/(exp(1/scale)+1)
#casts scale to Fraction
#assumes scale>=0
def sample_dlaplace(scale,rng=None):
    if rng is None:
        rng = random.SystemRandom()
    scale = Fraction(scale)
    assert scale >= 0
    while True:
        sign=sample_bernoulli(Fraction(1,2),rng)
        magnitude=sample_geometric_exp_fast(1/scale,rng)
        if sign==1 and magnitude==0: continue
        return magnitude*(1-2*sign)
        
#compute floor(sqrt(x)) exactly
#only requires comparisons between x and integer
def floorsqrt(x):
    assert x >= 0
    #a,b integers
    a=0 #maintain a^2<=x
    b=1 #maintain b^2>x
    while b*b <= x:
        b=2*b #double to get upper bound
    #now do binary search
    while a+1<b:
        c=(a+b)//2 #c=floor((a+b)/2)
        if c*c <= x:
            a=c
        else:
            b=c
    #check nothing funky happened
    #assert isinstance(a,int) #python 3
    #assert isinstance(a,(int,long)) #python 2
    return a
    
#sample from a discrete Gaussian distribution N_Z(0,sigma2)
#Returns integer x with Pr[x] = exp(-x^2/(2*sigma2))/normalizing_constant(sigma2)
#mean 0 variance ~= sigma2 for large sigma2
#casts sigma2 to Fraction
#assumes sigma2>=0
def sample_dgauss(sigma2,rng=None):
    if rng is None:
        rng = random.SystemRandom()
    sigma2=Fraction(sigma2)
    if sigma2==0: return 0 #degenerate case
    assert sigma2 > 0
    t = floorsqrt(sigma2)+1
    while True:
        candidate = sample_dlaplace(t,rng=rng)
        bias=((abs(candidate)-sigma2/t)**2)/(2*sigma2)
        if sample_bernoulli_exp(bias,rng)==1:
            return candidate
        
#########################################################################
#DONE That's it! Now some utilities

import math #need this, code below is no longer exact

#Compute the normalizing constant of the discrete gaussian
#i.e. sum_{x in Z} exp(-x^2/2sigma2)
#By Poisson summation formula, this is equivalent to
# sqrt{2*pi*sigma2}*sum_{y in Z} exp(-2*pi^2*sigma2*y^2)
#For small sigma2 the former converges faster
#For large sigma2, the latter converges faster
#crossover at sigma2=1/2*pi
#For intermediate sigma2, this code will compute both and check
def normalizing_constant(sigma2):
    original=None
    poisson=None
    if sigma2<=1:
        original = 0
        x=1000 #summation stops at exp(-x^2/2sigma2)<=exp(-500,000)
        while x>0:
            original = original + math.exp(-x*x/(2.0*sigma2))
            x = x - 1 #sum from small to large for improved accuracy
        original = 2*original + 1 #symmetrize and add x=0
    if sigma2*100 >= 1:
        poisson = 0
        y = 1000 #summation stops at exp(-y^2*2*pi^2*sigma2)<=exp(-190,000)
        while y>0:
            poisson = poisson + math.exp(-math.pi*math.pi*sigma2*2*y*y)
            y = y - 1 #sum from small to large
        poisson = math.sqrt(2*math.pi*sigma2)*(1+2*poisson)
    if poisson is None: return original
    if original is None: return poisson
    #if we have computed both, check equality
    scale = max(1,math.sqrt(2*math.pi*sigma2)) #tight-ish lower bound on constant
    assert -1e-15*scale <= original-poisson <= 1e-15*scale
    #10^-15 is about as much precision as we can expect from double precision floating point numbers
    #64-bit float has 56-bit mantissa 10^-15 ~= 2^-50
    return (original+poisson)/2

#compute the variance of discrete gaussian
#mean is zero, thus:
#var = sum_{x in Z} x^2*exp(-x^2/(2*sigma2)) / normalizing_constant(sigma2)
#By Poisson summation formula, we have equivalent expression:
# variance(sigma2) = sigma2 * (1 - 4*pi^2*sigma2*variance(1/(4*pi^2*sigma2)) )
#See lemma 20 https://arxiv.org/pdf/2004.00010v3.pdf#page=17
#alternative expression converges faster when sigma2 is large
#crossover point (in terms of convergence) is sigma2=1/(2*pi)
#for intermediate values of sigma2, we compute both expressions and check
def variance(sigma2):
    original=None
    poisson=None
    if sigma2<=1: #compute primary expression
        original=0
        x = 1000 #summation stops at exp(-x^2/2sigma2)<=exp(-500,000)
        while x>0: #sum from small to large for improved accuracy
            original = original + x*x*math.exp(-x*x/(2.0*sigma2))
            x=x-1
        original = 2*original/normalizing_constant(sigma2)
    if sigma2*100>=1:
        poisson=0 #we will compute sum_{y in Z} y^2 * exp(-2*pi^2*sigma2*y^2)
        y=1000 #summation stops at exp(-y^2*2*pi^2*sigma2)<=exp(-190,000)
        while y>0: #sum from small to large
            poisson = poisson + y*y*math.exp(-y*y*2*sigma2*math.pi*math.pi)
            y=y-1
        poisson = 2*poisson/normalizing_constant(1/(4*sigma2*math.pi*math.pi))
        #next convert from variance(1/(4*pi^2*sigma2)) to variance(sigma2)
        poisson = sigma2*(1-4*sigma2*poisson*math.pi*math.pi)
    if original is None: return poisson
    if poisson is None: return original
    #if we have computed both check equality
    assert -1e-15*sigma2 <= original-poisson <= 1e-15*sigma2
    return (original+poisson)/2

#########################################################################
#DONE Now some basic testing code

import matplotlib.pyplot as plt #only needed for testing
import time #only needed for testing

#This generates n samples from sample_dgauss(sigma2)
#It times this and releases statistics
#produces a histogram plot if plot==True
#if plot==None it will only produce a histogram if it's not too large
#can save image instead of displaying by specifying a path  e.g., save="plot.png"
def plot_histogram(sigma2,n,save=None,plot=None):
    #generate samples
    before=time.time()
    samples = [sample_dgauss(sigma2) for i in range(n)]
    after=time.time()
    print("generated "+str(n)+" samples in "+str(after-before)+" seconds ("+str(n/(after-before))+" samples per second) for sigma^2="+str(sigma2))
    #now process
    samples.sort()
    values=[]
    counts=[]
    counter=None
    prev=None
    for sample in samples:
        if prev is None: #initializing
            prev=sample
            counter=1
        elif sample==prev: #still same element
            counter=counter+1
        else:
            #add prev to histogram
            values.append(prev)
            counts.append(counter)
            #start counting
            prev=sample
            counter=1
    #add final value
    values.append(prev)
    counts.append(counter)
    
    #print & sum
    sum=0
    sumsquared=0
    kl=0 #compute KL divergence betwen empirical distribution and true distribution
    norm_const=normalizing_constant(sigma2)
    true_var=variance(sigma2)
    for i in range(len(values)):
        if len(values)<=100: #don't print too much
            print(str(values[i])+":\t"+str(counts[i]))
        sum = sum + values[i]*counts[i]
        sumsquared = sumsquared + values[i]*values[i]*counts[i]
        kl = kl + counts[i]*(math.log(counts[i]*norm_const/n)+values[i]*values[i]/(2.0*sigma2))
    mean = Fraction(sum,n)
    var=Fraction(sumsquared,n)
    kl=kl/n
    print("mean="+str(float(mean))+" (true=0)")
    print("variance="+str(float(var))+" (true="+str(true_var)+")")
    print("KL(empirical||true)="+str(kl)) # https://en.wikipedia.org/wiki/G-test
    assert kl>0 #kl divergence always >=0 and ==0 iff empirical==true, which is impossible
    #now plot
    if plot is None:
        plot = (len(values)<=1000) #don't plot if huge
    if not plot: return
    ideal_counts = [n*math.exp(-x*x/(2.0*sigma2))/norm_const for x in values]
    plt.bar(values, counts)
    plt.plot(values, ideal_counts,'r')
    plt.title("Histogram of samples from discrete Gaussian\nsigma^2="+str(sigma2)+" n="+str(n))
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.clf()

if __name__ == '__main__':
    print("This is the discrete Gaussian sampler")
    print("See the paper https://arxiv.org/abs/2004.00010")
    print("Now running some basic testing code")
    print("Start by calculating normalizing constant and variance for different values")
    #some test code for normalizing_constant and variance functions
    for sigma2 in [0.1**100,0.1**6,0.001,0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100,10**6,10**20,10**100]:
        #internal asserts do some testing when 0.01<=sigma2<=1
        c=normalizing_constant(sigma2)
        v=variance(sigma2)
        #print
        print("sigma^2="+str(sigma2) + ":\tnorm_const=" + str(c) + "=sqrt{2*pi}*sigma*" + str(c/math.sqrt(2*math.pi*sigma2)) + "\tvar=" + str(v))
    #print a few samples
    #for i in range(20): print sample_dgauss(1)
    #plot histogram and statistics
    #includes timing
    print("Now run the sampler")
    print("Start with very large sigma^2=10^100 -- for timing purposes only")
    plot_histogram(10**100,100000,plot=False) #large var, this will just be for timing
    print("Now sigma^2=10 -- will display a histogram")
    plot_histogram(10,100000) #small var, this will produce plot

