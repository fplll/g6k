******************************
The General Sieve Kernel (G6K)
******************************

.. image:: https://github.com/fplll/g6k/workflows/Tests/badge.svg
    :target: https://github.com/fplll/g6k/actions?query=workflow%3ATests

G6K is a C++ and Python library that implements several Sieve algorithms to be used in more advanced lattice reduction tasks. It follows the stateful machine framework from: 

Martin R. Albrecht and Léo Ducas and Gottfried Herold and Elena Kirshanova and Eamonn W. Postlethwaite and Marc Stevens, 
The General Sieve Kernel and New Records in Lattice Reduction.

The article is available `in this repository <https://github.com/fplll/g6k/blob/master/article.pdf>`__ and on `eprint <https://eprint.iacr.org/2019/089>`__ .


Building the library
====================

You will need the current master of FPyLLL. See ``bootstrap.sh`` for creating (almost) all dependencies from scratch:

.. code-block:: bash

    # once only: creates local python env, builds fplll, fpylll and G6K
    ./bootstrap.sh [ -j # ]
    
    # for every new shell: activates local python env
    source ./activate                   

On systems with co-existing python2 and 3, you can force a specific version installation using ``PYTHON=<pythoncmd> ./boostrap.sh`` instead.
The number of parallel compilation jobs can be controlled with `-j #`.

If building via ```./bootstrap.sh``` fails, then the script will return an error code. 
The error codes are documented in ```bootstrap.sh.```

Otherwise, you will need fplll and fpylll already installed and build the G6K Cython extension like so:

.. code-block:: bash

    pip install Cython
    pip install -r requirements.txt
    python setup.py build_ext --inplace [ -j # ]

This builds G6K **in place**. Alternatively, you can skip ```--inplace``` and run ```python setup.py install``` as usual after building.
    
It's possible to alter the C++ kernel build configuration as follows:

.. code-block:: bash

    make clean
    ./configure [opts...]           # e.g. opts: --enable-native --enable-templated-dim --with-max-sieving-dim=128
                                    # see ./configure --help for more options
    python setup.py build_ext [ -j # ]

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

    python ./full_sieve.py 100 --sieve hk3 --seed 23 --trials 2 --threads 26 --db_size_base 1.140174986570044 1.1414898159861084 1.1428031326523391 1.1441149417781413 1.14542524854309 1.146734058097168 1.1480413755610026 1.1493472060 1.153255825912013 1.154555758722808 1.1547005383

The whole experiment took ~15 h. If you do not want to wait that long, decrease the dimension. 
*Note* : Asymptotically, one would need to adjust the `saturation_radius` accordingly. However, at these dimensions, the default `db_size_factor` was large enough to accomodate saturation in practce.


Exact-SVP (Sec 6.1)
-------------------

Before benchmarking for exact-SVP, one must first determine the length of the shortest vector. To do
so on 3 lattices in each dimensions d ∈ {50, 52, 54, 56, 58}:

.. code-block:: bash

  python ./svp_exact_find_norm.py 50 -u 60 --workers 4 --challenge-seed 0 1 2

This will run 4 independent tasks in parrallel, and takes about 1 minute. Challenges will be
downloaded from https://www.latticechallenge.org/ if not already present.

Then, run and obtain averaged timing:

.. code-block:: bash

    python ./svp_exact.py 50 -u 60 --workers 3 --challenge-seed 0 1 2

Which will take around 10 seconds. To compare several algorithms, and average over 5 trials on each of the 3 lattices for d=50, you can run:

.. code-block:: bash

    python ./svp_exact.py 50 --workers 3 --trials 5 --challenge-seed 0 1 2 --svp/alg workout enum


SVP-challenge (Sec 6.2)
-----------------------

You can here run a single instance on multiple cores, for example:

.. code-block:: bash

    python ./svp_challenge.py 100 --threads 4

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


Other CLI programs and commands
===============================

It is also possible ot ask for HKZ reduction with hkz.py and hkz_maybe.py; the former really tries hard to get a HKZ basis (with no formal guarentees though) while the latter is providing something close to a HKZ basis significantly significantly faster than the former.

Other options:
Each of the parameters PARAM listed in g6k/siever_param.pyx can be set-up to a value VAL from the command line

.. code-block:: bash

        --PARAM VAL

Though some of them may be overwritten by the call chain. A subset of reasonable parameter to play with are:

.. code-block:: python

        threads                         # Number of threads collaborating in a single g6k instance. Default=1
        sample_by_sums                  # When increasing the db size, do that aggressively by sampling vectors as sums of existing vectors. Default=True
        otf_lift                        # Lift vectors on the fly; slower per sieve, but highter probability to find a short vector in the lift context. Default=True
        lift_radius                     # Bound (relative to squared-GH) to try to lift a vector on the fly. Default=1.7
        saturation_ratio                # Stop the sieve when this ratio of vector has been found compared to the expected number of vector. Default=.5 
        saturation_radius               # Define the ball square-radius for the saturation_ratio condition. Default=1.333333333
        dual_mode                       # Implicitly run all operations on the dual-basis (in reversed order).

Other parameters specific to subprograms SUBPRG∊{pump, workout, bkz} can be set-up to a value VAL form the CLI by adding the option

.. code-block:: bash

        --SUBPRG/PARAM VAL

One can also specify a set of values, or a range of value, to iterate over

.. code-block:: bash


        --SUBPRG/PARAM VAL0 VAL1 ... VALx
        --SUBPRG/PARAM MIN_VAL~MAX_VAL
        --SUBPRG/PARAM MIN_VAL~MAX_VAL~STEP_VAL

One can find all the available option by browsing through the programs in the g6k/algorithms/ subdirectory.

It is also possible to plot or to output the so called `profile', namely the logarithmic plot of the Gram-Schmidt norms, with the option

.. code-block:: bash

        --profile filename.csv      #exporting raw data as column seperated values
        --profile filename.EXT      #for EXT∊{png,pdf,...} plot in a file, requires matplotlib
        --profile show              #plot in a pop-up window, requires matplotlib


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

More examples can be found in the folder  ``examples``.

Acknowledgements
================

This project was supported through the European Union PROMETHEUS project (Horizon 2020 Research and Innovation Program, grant 780701), EPSRC grant EP/P009417/1 and EPSRC grant EP/S020330/1.
