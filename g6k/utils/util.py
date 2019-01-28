import os.path
import requests
import sys
import logging

from math import log

from fpylll import IntegerMatrix, LLL, FPLLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.bkz_stats import dummy_tracer, Accumulator
from fpylll import Enumeration, EnumerationError
from fpylll.util import gaussian_heuristic, set_random_seed
from g6k.utils.stats import SieveTreeTracer


def load_matrix_file(filepath, randomize=False, seed=None):
    """
    Load matrix from file, LLL reduce (and randomize).

    :param filepath: Load matrix from this file
    :param randomize: Randomize the basis
    :param seed: Seed for randomization
    :returns: lattice basis and BKZ object

    """
    A = IntegerMatrix.from_file(filepath)
    A = LLL.reduction(A)
    A = IntegerMatrix.from_matrix(A, int_type="long")

    M = GSO.Mat(A, float_type="double", flags=GSO.ROW_EXPO)
    bkz = BKZReduction(M)

    if seed is not None:
        FPLLL.set_random_seed(seed)

    if randomize:
        bkz.randomize_block(0, A.nrows, density=A.ncols/4)
        LLL.reduction(A)
        bkz = BKZReduction(A)

    LLL.reduction(A)
    bkz.lll_obj()  # to initialize bkz.M etc

    return A, bkz


def load_svpchallenge_and_randomize(n, s=0, seed=None, verbose=True):
    """
    Load SVP challenge (and randomize)

    :param n: dimension
    :param s: SVP challenge seed
    :param seed: seed for rerandomization
    :returns: lattice basis and BKZ object

    TESTS::

        >>> from g6k.utils.util import load_svpchallenge_and_randomize
        >>> # suppressing downloading message
        >>> print "skip from here ",; A, _ = load_svpchallenge_and_randomize(50) # doctest: +ELLIPSIS
        skip from here ...
        >>> B, _ = load_svpchallenge_and_randomize(50)
        >>> A == B
        False

        >>> A, _ = load_svpchallenge_and_randomize(50, seed=0)
        >>> B, _ = load_svpchallenge_and_randomize(50, seed=0)
        >>> A == B
        True

    """

    filename = "svpchallenge/svpchallenge-dim-%03d-seed-%02d.txt"%(n, s)

    if not os.path.isdir("svpchallenge"):
        os.mkdir("svpchallenge")

    if os.path.isfile(filename) is False:
        logging.info("Did not find '{filename}', downloading ...".format(filename=filename),)
        r = requests.post("https://www.latticechallenge.org/svp-challenge/generator.php",
                          data={'dimension': n, 'seed': s, 'sent': 'True'})
        logging.info("%s %s"%(r.status_code, r.reason))
        fn = open(filename, "w")
        fn.write(r.text)
        fn.close()

    return load_matrix_file(filename, randomize=True, seed=seed)


def load_prebkz(n, s=0, blocksize=40):
    """
    """

    filename = "qarychallenge/prebkz-%02d-dim-%03d-seed-%02d.txt"%(blocksize, n, s)

    if not os.path.isdir("qarychallenge"):
        os.mkdir("qarychallenge")

    if os.path.isfile(filename) is False:
        set_random_seed(s)
        A = IntegerMatrix.random(n, "qary", q=2**30, k=n//2)
        print "Did not find '{filename}'. Creating and reducing".format(filename=filename)
        print "created, ",
        sys.stdout.flush()
        A = LLL.reduction(A)
        print "LLLed, ",
        sys.stdout.flush()

        if A.nrows >= 160:
            float_type = "long double"
        elif A.nrows >= 200:
            float_type = "dd"
        else:
            float_type = "double"

        M = GSO.Mat(A, float_type=float_type, flags=GSO.ROW_EXPO)

        bkz = BKZReduction(M)

        for b in range(10, blocksize+1):
            print "\r created, LLLed, BKZed %d"%b,
            sys.stdout.flush()

            par = fplll_bkz.Param(b, strategies=fplll_bkz.DEFAULT_STRATEGY,
                                  max_loops=1, flags=fplll_bkz.MAX_LOOPS)
            bkz(par)

        print

        fn = open(filename, "w")
        fn.write(str(A))
        fn.close()

    return load_matrix_file(filename, randomize=False)


SVPCHALLENGE_NORM_FMT = "svpchallenge/svpchallenge-dim-%03d-seed-%02d.svp"


def load_svpchallenge_norm(n, s=0):
    filename = SVPCHALLENGE_NORM_FMT%(n, s)
    if os.path.isfile(filename) is False:
        print "Did not find '{filename}'. Please run svp_exact_find_norm for this instance first".format(
            filename=filename)
    with open(filename, 'r') as file:
        norm = float(file.read())
    return norm


def save_svpchallenge_norm(n, norm, s=0):
    filename = SVPCHALLENGE_NORM_FMT%(n, s)

    with open(filename, 'w') as fh:
        print >>fh, norm
    return


# Implement bkz2.svp_reduction with precise radius goal not success probability


def svp_reduction_until_goal(bkz, params, goal):
    n = bkz.M.d
    r = [bkz.M.get_r(i, i) for i in range(0, n)]
    gh = gaussian_heuristic(r)

    while bkz.M.get_r(0, 0) > goal:
        bkz.randomize_block(0, n)
        bkz.svp_preprocessing(0, n, params)

        strategy = params.strategies[n]
        radius = goal
        pruning = strategy.get_pruning(goal, gh)

        try:
            enum_obj = Enumeration(bkz.M)
            max_dist, solution = enum_obj.enumerate(
                0, n, radius, 0, pruning=pruning.coefficients)[0]
            bkz.svp_postprocessing(0, n, solution, tracer=dummy_tracer)
            # rerandomize = False

        except EnumerationError:
            # rerandomize = True
            pass

    bkz.lll_obj()
    return


def find_goal(dim, prelim_rep):
    # use traditional SVP solvers to find goal length
    params = fplll_bkz.Param(
        block_size=dim,
        max_loops=1,
        strategies=fplll_bkz.DEFAULT_STRATEGY,
        flags=fplll_bkz.GH_BND)
    A, bkz = load_svpchallenge_and_randomize(dim)
    gh = gaussian_heuristic([bkz.M.get_r(i, i) for i in range(dim)])
    goal = None
    for _ in range(prelim_rep):
        A, bkz = load_svpchallenge_and_randomize(dim)
        bkz.svp_reduction(0, dim, params)
        r0 = bkz.M.get_r(0, 0)
        if goal is None:
            goal = 1.003 * r0
        else:
            goal = min(goal, 1.001 * r0)
    return goal, gh


def run_it(p, f, A, prefix=""):
    r = []
    for _, retval in enumerate(p.imap_unordered(f, A, 1)):
        r.append(retval)
    return r


def db_stats(stats):
    """
    Given a list of traces, find the average of the maximum |db| and the
    maximum of the maximum |db| for the traces

    :param stats: a list of traces of type ``Node``

    """

    max_dbs = Accumulator(0, repr="avg", count=False)
    for stat in stats:
        max_dbs += stat.accumulate("|db|",
                                   filter=lambda node: SieveTreeTracer.is_sieve_node(node.label),
                                   repr="max").max

    return log(max_dbs.avg, 2), log(max_dbs.max, 2)


def load_lwe_challenge(n=40, alpha=0.005):
    """
    Load LWE challenge from file or website.

    :param n: LWE dimension
    :param alpha: the *standard deviation* of the secret is alpha*q

    """
    alpha = int(round(alpha * 1000))
    start = "lwechallenge"

    if not os.path.isdir(start):
        os.mkdir(start)

    end = "{n:03d}-{alpha:03d}-challenge.txt".format(n=n, alpha=alpha)
    filename = os.path.join(start, end)
    if not os.path.isfile(filename):
        url = ("https://www.latticechallenge.org/lwe_challenge/challenges/"
               "LWE_{n:d}_{alpha:03d}.txt")
        url = url.format(n=n, alpha=alpha)
        r = requests.get(url)
        m = "Cannot retrieve challenge; server response was: '%s'. \n URL was: %s" % (r.reason, url)
        if not r.status_code == 200:
            raise ValueError(m)
        fn = open(filename, "w")
        fn.write(r.text)
        fn.close()

    data = open(filename, "r").readlines()
    n, m, q = map(lambda x: int(x), [data[0], data[1], data[2]])

    c_index = 3 if data[3].startswith("[") else 4

    A = eval(",".join([s_.replace(" ", ", ") for s_ in data[c_index+1:]]))
    A = IntegerMatrix.from_matrix(A)
    c = tuple(eval(data[c_index].replace(" ", ", ")))
    return A, c, q
