# -*- coding: utf-8 -*-
"""
Command Line Interfaces
"""
from __future__ import absolute_import
from __future__ import print_function
import argparse
import copy
import datetime
import logging
import multiprocessing_logging
import os
import re
import socket
import subprocess
import sys
from collections import OrderedDict
from multiprocessing import Pool

from fpylll import BKZ

from g6k.siever_params import SieverParams
import six
from six.moves import range


cli_arg_aliases = {
    "--wo/": "--workout/",
    "--sat_": "--saturation_",
    "--sat-": "--saturation-",
    "--chal_": "--challenge_",
    "--sieve": "--default_sieve",
    "hk3": "gauss_triple_mt",
    "d4f": "dim4free",
    "pnj": "pump_and_jump",
    "beta": "blocksize"
}


def apply_aliases(cli_args):
    """
    Apply aliases to command line argument.

        >>> apply_aliases(['--wo/bar', '4', '--sat_grumpf', '1.3'])
        ['--workout/bar', '4', '--saturation_grumpf', '1.3']

    :param cli_args: list of strings

    """
    acli_args = []

    for arg in cli_args:
        for x, y in six.iteritems(cli_arg_aliases):
            arg = arg.replace(x, y)
        acli_args.append(arg)

    return acli_args


def pop_prefixed_params(prefix, params):
    """
    pop all parameters from ``params`` with a prefix.

    A prefix is any string before the first "/" in a string::

        >>> pop_prefixed_params('foo', {'foo/bar': 1, 'whoosh': 2})
        {'bar': 1}

    :param prefix: prefix string
    :param params: key-value store where keys are strings

    """
    keys = [k for k in params]
    poped_params = {}
    if not prefix.endswith("/"):
        prefix += "/"

    for key in keys:
        if key.startswith(prefix):
            poped_key = key[len(prefix):]
            poped_params[poped_key] = params.pop(key)

    return poped_params


def run_all(f, params_list, lower_bound=40, upper_bound=0, step_size=2, trials=1, workers=1, pickle=False, seed=0):
    """Call ``f`` on matrices with dimensions in ``range(lower_bound, upper_bound, step_size)``

    :param params_list: run ``f`` for all parameters given in ``params_list``
    :param lower_bound: lowest lattice dimension to consider (inclusive)
    :param upper_bound: upper bound on lattice dimension to consider (exclusive)
    :param step_size: increment lattice dimension in these steps
    :param trials: number of experiments to run per dimension
    :param workers: number of parallel experiments to run
    :param pickle: pickle statistics
    :param seed: randomness seed

    """
    if upper_bound == 0:
        upper_bound = lower_bound + 1

    jobs, stats = [], OrderedDict()
    for n in range(lower_bound, upper_bound, step_size):
        for params in params_list:
            stats[(n, params)] = []
            for t in range(trials):
                args = (n, params, seed+t)
                jobs.append(args)

    if workers == 1:
        for job in jobs:
            n, params, seed_ = job
            res = f(copy.deepcopy(job))
            stats[(n, params)].append(res)
            logging.debug(res)

    else:
        pool = Pool(workers)
        for i, res in enumerate(pool.map(f, jobs)):
            n, params, seed_ = jobs[i]
            stats[(n, params)].append(res)
            logging.debug(res)

    return stats


def git_revisionf():
    git_revision = []
    cmds = [("git", "show", "-s", "--format=%cd", "HEAD", "--date=short"),
            ("git", "rev-parse", "--abbrev-ref", "HEAD"),
            ("git", "show", "-s", "--format=%h", "HEAD", "--date=short")]

    for cmd in cmds:
        try:
            r = str(subprocess.check_output(cmd, stderr=subprocess.STDOUT).rstrip())
            git_revision.append(r)
        except (ValueError, subprocess.CalledProcessError):
            pass

    git_revision = "-".join(git_revision)
    return git_revision


def log_filenamef():
    base = os.path.basename(sys.argv[0]).replace(".py", "")
    revision = git_revisionf()
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    hostname = socket.gethostname()
    log_filename = "{base},{hostname},{date},{revision}.log".format(base=base,
                                                                    hostname=hostname,
                                                                    date=date,
                                                                    revision=revision)
    log_filename = os.path.join("logs", log_filename)
    return log_filename


def parse_args(description, ParamsClass=SieverParams, **kwds):
    """
    Parse command line arguments.

    The command line parser accepts the standard parameters as printed by calling it with
    ``--help``.  All other parameters are used to construct params objects.  For example.

    ./foo 80 --workers 4 --trials 2 -S 1337 --a 1 2 - b 3 4

    would operate on dimension 80 with parameters (a: 1, b: 3), (a: 1, b: 4), (a: 2, b: 3), (a: 2,
    b: 4), i.e. the Cartesian product of all parameters.  It will run two trials each using four
    workers. Note that each worker may use several threads, too. The starting seed is `1337`.

    :param description: help message
    :param kwds: default parameters

    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lower_bound', type=int,
                        help="lowest lattice dimension to consider (inclusive)")
    parser.add_argument('-u', '--upper-bound', type=int, dest="upper_bound", default=0,
                        help="upper bound on lattice dimension to consider (exclusive)")
    parser.add_argument('-s', '--step-size', type=int, dest="step_size", default=2,
                        help="increment lattice dimension in these steps")
    parser.add_argument('-t', '--trials', type=int, dest="trials", default=1,
                        help="number of experiments to run per dimension")
    parser.add_argument('-w', '--workers', type=int, dest="workers", default=1,
                        help="number of parallel experiments to run")
    parser.add_argument('-p', '--pickle', action='store_true', dest="pickle",
                        help="pickle statistics")
    parser.add_argument('-S', '--seed',  type=int, dest="seed", default=0,
                        help="randomness seed")
    parser.add_argument('--dry-run', dest="dry_run", action='store_true',
                        help="Show parameters that would be used but don't run any actual experiments.")
    parser.add_argument('--show-defaults', dest="show_defaults", action='store_true',
                        help="Show default parameters and exit.")
    parser.add_argument('--loglvl', type=str, help="Logging level (one of DEBUG, WARN, INFO)", default="INFO")
    parser.add_argument('--log-filename', dest="log_filename", type=str, help="Logfile filename", default=None)
    parser.add_argument('--profile', dest="profile", type=str, help="Output final log-profile into specified file (.csv, .pdf, .png, ...)", default=None)
    args, unknown = parser.parse_known_args()

    kwds_ = OrderedDict()
    for k, v in six.iteritems(kwds):
        k_ = k.replace("__", "/")
        kwds_[k_] = v
    kwds = kwds_

    if args.show_defaults:
        pp = ParamsClass(**kwds)
        slen = max(len(p) for p in pp) + 1
        fmt = "{key:%ds}: {value}"%slen
        for k, v in six.iteritems(pp):
            print(fmt.format(key=k, value=v))
        exit(0)

    all_params = OrderedDict([("", ParamsClass(**kwds))])

    unknown_args = OrderedDict()
    unknown = apply_aliases(unknown)

    # NOTE: This seems like the kind of thing the standard library can do (better)
    i = 0
    while i < len(unknown):
        k = unknown[i]
        if not (k.startswith("--") or k.startswith("-")):
            raise ValueError("Failure to parse command line argument '%s'"%k)
        k = re.match("^-+(.*)", k).groups()[0]
        k = k.replace("-", "_")
        unknown_args[k] = []
        i += 1
        for i in range(i, len(unknown)):
            v = unknown[i]
            if v.startswith("--") or v.startswith("-"):
                i -= 1
                break

            try:
                L = re.match("([0-9]+)~([0-9]+)~?([0-9]+)?", v).groups()
                if L[2] is not None:
                    v = range(int(L[0]), int(L[1]), int(L[2]))
                else:
                    v = range(int(L[0]), int(L[1]))
                unknown_args[k].extend(v)
                continue
            except:
                pass

            try:
                v = eval(v, {"BKZ": BKZ})
            except NameError:
                v = v
            except SyntaxError:
                v = v
            if not isinstance(v, (list, tuple)):
                v = [v]
            unknown_args[k].extend(v)
        i += 1
        if not unknown_args[k]:
            unknown_args[k] = [True]

    for k, v in six.iteritems(unknown_args):
        all_params_ = OrderedDict()
        for p in all_params:
            for v_ in v:
                p_ = copy.copy(all_params[p])
                p_[k] = v_
                all_params_[p+"'%s': %s, "%(k, v_)] = p_
        all_params = all_params_

    log_filename = args.log_filename
    if log_filename is None:
        log_filename = log_filenamef()

    multiprocessing_logging.install_mp_handler()

    if not os.path.isdir("logs"):
        os.makedirs("logs")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)5s:%(name)12s:%(asctime)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S %Z',
                        filename=log_filename)

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, args.loglvl.upper()))
    console.setFormatter(logging.Formatter('%(name)s: %(message)s',))
    logging.getLogger('').addHandler(console)

    if args.dry_run:
        for params in all_params:
            print(params)
        exit(0)

    return args, all_params
