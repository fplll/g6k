******************************
The General Sieve Kernel (G6K)
******************************

.. image:: https://travis-ci.org/fplll/g6k.svg?branch=master
    :target: https://travis-ci.org/fplll/g6k

G6K is a C++ and Python library that implements several Sieve algorithms to be used in more advanced lattice reduction tasks. It follows the stateful machine framework from: 

Martin R. Albrecht and Léo Ducas and Gottfried Herold and Elena Kirshanova and Eamonn W. Postlethwaite and Marc Stevens, 
The General Sieve Kernel and New Records in Lattice Reduction.

The article is available `in this repository <https://github.com/fplll/g6k/blob/master/article.pdf>`__ and on `eprint <https://eprint.iacr.org/2019/089>`__ .


Building the library
====================

You will need the current master of FPyLLL. See ``bootstrap.sh`` for creating (almost) all dependencies from scratch:

.. code-block:: bash

    ./bootstrap.sh                # once only: creates local python env, builds fplll, fpylll and G6K
    source ./activate             # for every new shell: activates local python env
    ./rebuild.sh -f               # whenever you want to rebuild G6K

Otherwise, you will need fplll and fpylll already installed and build the G6K Cython extension **in place** like so:

.. code-block:: bash

    pip install Cython
    pip install -r requirements.txt
    ./rebuild.sh -f

Remove ``-f`` option to compile faster (fewer optimisations). See ``rebuild.sh`` for more options.


Tests
=====

.. code-block:: bash

    python -m pytest


Gathering test coverage
-----------------------

Uncomment the line ``extra_compile_args += ["-DCYTHON_TRACE=1"]`` in ``setup py.`` and recompile. Then run

.. code-block:: bash

    py.test --cov=g6k


Reproducing experiments of the paper for the command line
=========================================================

3-sieve (Sec 5.1)
-----------------

To recreate Figure 2, run (if you have 26 threads, otherwise change ``--threads`` and, possibly, decrease the dimension):

.. code-block:: bash

    ./full_sieve.py 100 --sieve gauss_triple_mt --seed 23 --trials 2 --threads 26 --db_size_base 1.140174986570044 1.1414898159861084 1.1428031326523391 1.1441149417781413 1.14542524854309 1.146734058097168 1.1480413755610026 1.1493472060 1.153255825912013 1.154555758722808 1.1547005383

The whole experiment took ~15 h. If you do not want to wait that long, decrease the dimension.


Exact-SVP (Sec 6.1)
-------------------

Before benchmarking for exact-SVP, one must first determine the length of the shortest vector. To do
so on 3 lattices in each dimensions d ∈ {50, 52, 54, 56, 58}:

.. code-block:: bash

  ./svp_exact_find_norm.py 50 -u 60 --workers 4 --challenge-seed 0 1 2

This will run 4 independent tasks in parrallel, and takes about 1 minute. Challenges will be
downloaded from https://www.latticechallenge.org/ if not already present.

Then, run and obtain averaged timing:

.. code-block:: bash

    ./svp_exact.py 50 -u 60 --workers 3 --challenge-seed 0 1 2

Which will take around 10 seconds. To compare several algorithms, and average over 5 trials on each of the 3 lattices for d=50, you can run:

.. code-block:: bash

    ./svp_exact.py 50 --workers 3 --trials 5 --challenge-seed 0 1 2 --sieve gauss bgj1 enum


SVP-challenge (Sec 6.2)
-----------------------

You can here run a single instance on multiple cores, for example:

.. code-block:: bash

    ./svp_challenge.py 100 --threads 4

The above may take between half a minute and 10 minutes depending on how lucky you are


BKZ (Sec 6.3)
-------------

To recreate the experiments in the paper run:

.. code-block:: bash

    python bkz.py 180 --bkz/betas 60:95:1 --bkz/pre_beta 59 --trials 8 --workers 8
    python bkz.py 180 --bkz/betas 60:93:1 --bkz/pre_beta 59 --trials 8 --workers 8 --bkz/extra_d4f 12
    python bkz.py 180 --bkz/betas 60:97:1 --bkz/pre_beta 59 --trials 8 --workers 8 --bkz/extra_d4f 12 --bkz/jump 3
    python bkz.py 180 --bkz/betas 60:85:1 --bkz/pre_beta 59 --trials 8 --workers 8 --bkz/alg naive
    python bkz.py 180 --bkz/betas 60:82:1 --bkz/pre_beta 59 --trials 8 --workers 8 --bkz/alg fpylll


LWE (Sec 6.4)
-------------

To automatically attempt to solve a Darmstadt LWE Challenge (n, alpha) run:

.. code-block:: bash

    python lwe_challenge.py n --lwe/alpha alpha


Interactive use of G6K from Python
==================================

General Sieving Kernel. We start by importing the siever and FPYLLL

.. code-block:: python

    >>> from fpylll import IntegerMatrix, LLL, FPLLL
    >>> from g6k import Siever

Construct a challenge instance

.. code-block:: python

    >>> FPLLL.set_random_seed(0x1337)
    >>> A = IntegerMatrix.random(50, "qary", k=25, bits=20)
    >>> A = LLL.reduction(A)

Construct the instance

.. code-block:: python

    >>> g6k = Siever(A)
    >>> g6k.initialize_local(0, 0, 50)
    >>> g6k(alg="gauss")

We recover the shortest vector found. Best lift returns the index, the squared norm and the vector expressed in base `A`:

.. code-block:: python

    >>> i, norm, coeffs = g6k.best_lifts()[0]
    >>> l = int(round(norm))
    >>> l < 3710000
    True

To test the answer we compute:

.. code-block:: python

    >>> v = A.multiply_left(coeffs)
    >>> sum(v_**2 for v_ in v) == l
    True

