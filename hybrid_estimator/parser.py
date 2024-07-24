from sys import argv
from math import sqrt

help_msg ="""Usage: python <kyber> -kappa=<kappa>
Parameters: <kyber>, kyber{160,176,192,208,224,240,256,512,768,1024}
<kappa>, integer >= 15
"""

def st_dev_central_binomial(eta):
    return sqrt(eta / 2.0)

#Central Binomial probablity mass functions
CB2 = [(1/2)**5, 5*(1/2)**5, 10*(1/2)**5, 10*(1/2)**5, 5*(1/2)**5, (1/2)**5 ]
CB3 = [(1/2)**7, 7*(1/2)**7, 21*(1/2)**7, 35*(1/2)**7, 35*(1/2)**7, 21*(1/2)**7, 7*(1/2)**7, (1/2)**7]

# Kyber512 = {'n': 2*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3}
# Kyber768 = {'n': 3*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}
# Kyber1024 = {'n': 4*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}

kyber_instances = {
    "kyber160" : {'n': 160, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber176" : {'n': 176, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber192" : {'n': 192, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber208" : {'n': 208, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber224" : {'n': 224, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber240" : {'n': 240, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber256" : {'n': 256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber512" : {'n': 2*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber768" : {'n': 3*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2},
    "kyber1024": {'n': 4*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}
}

def parse_all():
    if "-h" in argv:
        print(help_msg)
    else:
        if any( "-n=" in tmp for tmp in argv ):
            raise NotImplementedError("Only kyber{160,176,192,208,224,240,256,512,768,1024} instances are currently supported.")
            assert any( "-q=" in tmp for tmp in argv ), "If -n= flag is present, -q= flag is required."
            assert any( "-kappa=" in tmp for tmp in argv ), "If -n= flag is present, -q= flag is required."

            brk = 0
            for s in argv[1:]: #TODO: std_dev and dist
                if "-n=" in s:
                    # print("n found")
                    n = 2*int(s[3:]) #the notions of n in "kybern" and in the code differ by a factor of 2
                    brk += 1
                    continue

                if "-q=" in s:
                    # print("q found")
                    q = int(s[3:])
                    brk += 1
                    continue

                if "-kappa=" in s:
                    # print("kappa found")
                    kappa = int(s[7:])
                    brk += 1
                    continue
                if brk >= 3:
                    break

            assert brk>= 3, "-n, -q or -kappa flag is not provided."
        else:
            assert any( "kyber" in tmp.lower() for tmp in argv ), "If -n= flag is present, -q= flag is required."
            brk = 0
            for s in argv:
                if "kyber" in s.lower():
                    # print("kyber found")
                    try:
                        KyberParam = kyber_instances[s.lower()]
                    except KeyError:
                        raise Exception(f"Kyber instance {s.lower()} not supported.")
                    n, q = KyberParam['n'], KyberParam['q']
                    st_dev_e, dist = KyberParam['st_dev_e'], KyberParam['dist']
                    brk += 1
                    continue

                if "-kappa=" in s:
                    # print("kappa found")
                    kappa = int(s[7:])
                    brk += 1
                    continue
                if brk >= 2:
                    break

            assert brk>=2, "kyber or -kappa flag is not provided."

        return n, q, kappa, st_dev_e, dist

if __name__=="__main__":
    n, q, kappa, st_dev_e, dist = parse_all()
    print( n, q, kappa, st_dev_e, dist )
