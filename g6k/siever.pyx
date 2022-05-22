#cython: linetrace=True
"""
Generalised Sieving Kernel (G6K) Siever

This class is the interface to the C++ implementation of sieving algorithm.  All higher-level
algorithms go through this class.
"""

from fpylll import FPLLL
from fpylll.tools.bkz_stats import dummy_tracer
from cysignals.signals cimport sig_on, sig_off
from libcpp cimport bool
from libcpp.string cimport string
import warnings
import logging
import copy

from numpy import zeros, float64, int64, matrix, array, where, matmul, identity, dot

import numpy as npp
cimport numpy as np
from fpylll import LLL, GSO, IntegerMatrix
from math import ceil, floor

from decl cimport CompressedEntry, Entry
from decl cimport show_cpu_stats
from decl cimport MAX_SIEVING_DIM

from scipy.special import betaincinv

from siever_params import temp_params

from libc.math cimport NAN

class SaturationError(RuntimeError):
    pass

cdef class Siever(object):
    """
    Main class for the Generalized Sieving Kernel (G6K)
    """

    def __init__(self, M, SieverParams params=None, seed=None):
        """
        Initialize a new sieving object.

        :param M: a ``MatGSO`` object or an ``IntegerMatrix``
        :param seed: seed for random number generator (if ``None`` then FPLLL's rng is called)

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, GSO
            >>> from g6k import Siever
            >>> A = IntegerMatrix.random(50, "qary", k=25, bits=10)
            >>> _ = Siever(A, seed=0x1337)

        TESTS:

        If a ``MatGSo`` object is passed, it must have ``transform_enabled`` and
        ``inverse_transform_enabled`` set to ``True``.  These are used and may be reset to the
        identity internally::

            >>> _ = Siever(GSO.Mat(A))
            Traceback (most recent call last):
            ...
            ValueError: Siever requires UinvT enabled

            >>> M = GSO.Mat(A, U=IntegerMatrix.identity(50), UinvT=IntegerMatrix.identity(50))
            >>> _ = Siever(M)

        """

        if isinstance(M, GSO.Mat):
            if not (M.inverse_transform_enabled and M.UinvT.nrows == M.d):
                raise ValueError("Siever requires UinvT enabled")

        elif isinstance(M, IntegerMatrix):
            if M.nrows >= 200:
                float_type = "dd"
            elif M.nrows >= 160:
                float_type = "long double"
            else:
                float_type = "double"

            M = self.MatGSO(M, float_type=float_type)

        else:
            raise TypeError("Matrix must be IntegerMatrix or GSO object but got type '%s'"%type(M))

        if params is None:
            params = SieverParams()

        if not params.lift_unitary_only:
            raise NotImplementedError("`unitary_only=False` is not implemented yet.")

        self.M = M

        if seed is None:
            # access FPLLL's rng
            seed = IntegerMatrix.random(1, "uniform", bits=32)[0, 0]
        self._core = new Siever_c(params._core, <unsigned long>seed)
        self.params = copy.copy(params)

        self._core.full_n = M.d

        if self._core.full_n > self.max_sieving_dim:
            warnings.warn("Dimension of lattice is larger than maximum supported. To fix this warning, change the value of MAX_SIEVING_DIM in siever.h and recompile.")

        self.lll(0, M.d)
        self.initialized = False

    @classmethod
    def MatGSO(cls, A, float_type="d"):
        """
        Create a GSO object from `A` that works for G6K.

        :param A: an integer matrix
        :param float_type:  a floating point type or a precision (which implies MPFR)

        """
        try:
            float_type = int(float_type)
            FPLLL.set_precision(float_type)
            float_type = "mpfr"
        except (TypeError, ValueError):
            pass
        M = GSO.Mat(A, float_type=float_type,
                    U=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                    UinvT=IntegerMatrix.identity(A.nrows, int_type=A.int_type))
        return M

    @property
    def params(self):
        """
        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> A = IntegerMatrix.random(50, "qary", k=25, bits=10)
            >>> siever = Siever(A, seed=0x1337)
            >>> siever.params
            SieverParams({})

            >>> siever.params.reserved_n = 10
            Traceback (most recent call last):
            ...
            ValueError: This object is read only, create a copy to edit.

            >>> sp = siever.params.new()
            >>> sp.reserved_n = 10
            >>> siever.params = sp

        """
        return self._params

    @params.setter
    def params(self, SieverParams params):
        for k in params.unknown_attributes:
            warnings.warn("Attribute '%s' unknown"%k)
        self._params = params
        self._core.set_params(self._params._core)
        self._params.set_read_only()

    temp_params = temp_params

    def __dealloc__(self):
        del self._core

    def update_gso(self, int l_bound, int r_bound):
        """
        Update the Gram-Schmidt vectors (from the left bound l_bound up to the right bound r_bound).
        If in dual mode, l_bound and r_bound are automatically reflected about self.full_n/2.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> A = IntegerMatrix.random(50, "qary", k=25, bits=10)
            >>> siever = Siever(A, seed=0x1337)
            >>> siever.update_gso(0, 50)

        ..  warning:: Do not call ``self.M.update_gso()`` directly but call this function instead as
        this function also updates the sieve's internal GSO coefficients.  Otherwise, the two
        objects might get out of sync and behaviour becomes undefined.

        """
        if not (0 <= l_bound and l_bound <= r_bound and r_bound <= self.M.d):
            raise ValueError("Parameters %d, %d, %d do not satisfy constraint  0 <= l_bound <= r_bound <= self.M.d"%(l_bound, r_bound))


        cdef int i, j, k
        cdef int m = self.full_n
        cdef int n = r_bound - l_bound
        cdef int d = self.M.d

        if not self.params.dual_mode:
            for i in range(r_bound):
                self.M.update_gso_row(i, i)
        else:
            for i in range(m - l_bound):
                self.M.update_gso_row(i, i)

        cdef np.ndarray _mu = zeros((d, d), dtype=float64)
        cdef double[:,:] _mu_view = _mu
        cdef np.ndarray _rr = zeros(d, dtype=float64)
        cdef np.ndarray _muinv = identity(n, dtype=float64)
        cdef double[:,:] _muinv_view = _muinv

        if not self.params.dual_mode:
            # copy mu and rr from the MatGSO object
            for i in range(l_bound, r_bound):
                _rr[i] = self.M.get_r(i, i)
                _mu_view[i][i] = 1.
                for j in range(l_bound, i):
                    _mu_view[i][j] = self.M.get_mu(i, j)

        else:
            # copy mu and rr from the MatGSO object and invert them (to compute the GSO of the dual)
            for i in range(l_bound, r_bound):
                _rr[i] = 1. / self.M.get_r(m - 1 - i, m - 1 - i)
                _mu_view[i][i] = 1.
                for j in range(l_bound, i):
                    _mu_view[i][j] = self.M.get_mu(m - 1 - j, m - 1 - i)

            # the following inverts _mu (into _mu_view) by exploiting the lower triangular structure
            for i in range(n):
                for k in range(i+1, n):
                    _muinv_view[k, i] = -_mu_view[l_bound + k, l_bound + i]

                for k in range(i):
                    for j in range(i+1, n):
                        _muinv_view[j, k] += _muinv_view[j, i]*_muinv_view[i, k]

            _mu[l_bound:r_bound, l_bound:r_bound] = _muinv

        for i in range(l_bound, r_bound):
            _mu_view[i][i] = _rr[i]

        sig_on()
        self._core.load_gso(self.M.d, <double*>_mu.data)
        sig_off()

    def initialize_local(self, ll, l, r, update_gso=True):
        """
        Local set-up.

        - update the local GSO
        - recompute ``gaussian_heuristic`` for renormalization
        - reset ``best_lift_so_far`` if r changed
        - reset compression and uid functions


        :param ll: lift context left index (inclusive)
        :param l: sieve context left index (inclusive)
        :param r: sieve context right index (exclusive)

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> A = IntegerMatrix.random(50, "qary", k=25, bits=10)
            >>> siever = Siever(A, seed=0x1337)
            >>> siever.initialize_local(0, 0, 50)
            >>> siever.initialize_local(1, 1, 25)

        TESTS::

            >>> siever.initialize_local(5, 4, 25)
            Traceback (most recent call last):
            ...
            ValueError: Parameters 5, 4, 25 do not satisfy constraint  0 <= ll <= l <= r <= self.M.d

            >>> siever.initialize_local(0, 0, 51)
            Traceback (most recent call last):
            ...
            ValueError: Parameters 0, 0, 51 do not satisfy constraint  0 <= ll <= l <= r <= self.M.d

            >>> siever.initialize_local(-1, 0, 50)
            Traceback (most recent call last):
            ...
            ValueError: Parameters -1, 0, 50 do not satisfy constraint  0 <= ll <= l <= r <= self.M.d


        """
        if not (0 <= ll and ll <= l and l <= r and r <= self.M.d):
            raise ValueError("Parameters %d, %d, %d do not satisfy constraint  0 <= ll <= l <= r <= self.M.d"%(ll, l, r))

        if update_gso:
            self.update_gso(ll, r)
        sig_on()
        self._core.initialize_local(ll, l, r)
        sig_off()
        self.initialized = True

    @property
    def max_sieving_dim(self):
        """
        The maximum sieving dimension that's supported in this build of G6K.
        This value can be changed in the following ways:

            - Manually. You can simply change the ``MAX_SIEVING_DIM`` macro in siever.h and then
              recompile.

            - Automatically. You can change this value by supplying the
              ``--with-max-sieving-dim <dim>`` flag to ``./configure``, where ``<dim>`` is the
              maximum supported dimension. For nicer support with AVX/vectorisation, we recommend
              a multiple of 32. This will recompile the g6k kernel.

        EXAMPLE::
            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337).max_sieving_dim
            128

        """
        return MAX_SIEVING_DIM


    @property
    def full_n(self):
        """
        Full dimension of the lattice.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337).full_n
            50

        """
        return self._core.full_n

    @property
    def l(self):
        """
        Current lower bound.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> siever= Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337)
            >>> siever.initialize_local(0, 1, 11)
            >>> siever.l
            1

        """
        return self._core.l

    @property
    def r(self):
        """
        Current upper bound.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> siever= Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337)
            >>> siever.initialize_local(0, 1, 11)
            >>> siever.r
            11

        """
        return self._core.r

    @property
    def ll(self):
        """
        Current lift left bound.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> siever= Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337)
            >>> siever.initialize_local(0, 1, 11)
            >>> siever.r
            11

        """
        return self._core.ll


    @property
    def n(self):
        """
        Current dimension (r-l).

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> siever= Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337)
            >>> siever.initialize_local(0, 1, 11)
            >>> siever.r - siever.l == siever.n
            True

        """
        return self._core.n

    def __len__(self):
        """
        Return the number of vectors in the (compressed) database.

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(50, "qary", k=25, bits=10))
            >>> g6k = Siever(A, seed=0x1337)
            >>> g6k.initialize_local(0, 0, 50)
            >>> g6k(alg="gauss") # Run that first to avoid rank-loss bug
            >>> g6k()
            >>> len(g6k)
            4252

        ..  note :: The full database may contain more spurious vectors which will be removed the
                    next time the database is updated.

        """
        return self._core.cdb.size()

    def db_size(self, compressed=True):
        """
        Return the size of the database:

        :param compressed: return size of compressed database if ``True``

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(50, "qary", k=25, bits=10))
            >>> g6k = Siever(A, seed=0x1337)
            >>> g6k.initialize_local(0, 0, 50)
            >>> g6k(alg="gauss") # Run that first to avoid rank-loss bug
            >>> g6k()
            >>> g6k.db_size()
            4252
            >>> g6k.db_size(False)
            4252

        ..  note :: Returning the size of the non-compressed database is only useful for
                    debugging purposes, the user usually wants the compressed database.

        """
        return self._core.db_size()

    def itervalues(self):
        """
        Iterate over all entries in the database (in the order determined by the compressed database).

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 0, 10)
            >>> g6k()
            >>> len(g6k)
            20
            >>> db = list(g6k.itervalues())
            >>> out = db[0]; out if db[0][0] > 0 else tuple([-x for x in out])
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        We get coordinates wrt the basis B

        .. note:: This function is mainly used for debugging purposes.

        """
        cdef Entry *e;

        for i in range(self._core.cdb.size()):
            e = &self._core.db[self._core.cdb[i].i]
            r = [e.x[j] for j in range(self._core.n)]
            yield tuple(r)


    def reset_stats(self):
        self._core.reset_stats()
    ############# New statistics ############

    # This exports _core.statistics to python. Note that stats(self) and get_stat(self, name)
    # provide much better interfaces. For an overview of what these mean, see the dictionary / table
    # below

    # NOTE generated by scripts/genexporters.sh, hence no docstring.
    # NOTE _stat_get_foo returns a value, _stat_c_foo returns whether foo was collected
    #      if _stat_c_foo is False, _stat_get_foo is meaningless

    @property
    def _stat_get_reds_total(self):
        return self._core.statistics.get_stats_reds_total()

    @property
    def _stat_c_reds_total(self):
        return self._core.statistics.collect_statistics_reds_total

    @property
    def _stat_get_2reds_total(self):
        return self._core.statistics.get_stats_2reds_total()

    @property
    def _stat_c_2reds_total(self):
        return self._core.statistics.collect_statistics_2reds_total

    @property
    def _stat_get_3reds(self):
        return self._core.statistics.get_stats_3reds()

    @property
    def _stat_c_3reds(self):
        return self._core.statistics.collect_statistics_3reds

    @property
    def _stat_get_2reds_inner(self):
        return self._core.statistics.get_stats_2reds_inner()

    @property
    def _stat_c_2reds_inner(self):
        return self._core.statistics.collect_statistics_2reds_inner

    @property
    def _stat_get_2reds_outer(self):
        return self._core.statistics.get_stats_2reds_outer()

    @property
    def _stat_c_2reds_outer(self):
        return self._core.statistics.collect_statistics_2reds_outer

    @property
    def _stat_get_xorpopcnt_total(self):
        return self._core.statistics.get_stats_xorpopcnt_total()

    @property
    def _stat_c_xorpopcnt_total(self):
        return self._core.statistics.collect_statistics_xorpopcnt_total

    @property
    def _stat_get_xorpopcnt_inner(self):
        return self._core.statistics.get_stats_xorpopcnt_inner()

    @property
    def _stat_c_xorpopcnt_inner(self):
        return self._core.statistics.collect_statistics_xorpopcnt_inner

    @property
    def _stat_get_xorpopcnt_outer(self):
        return self._core.statistics.get_stats_xorpopcnt_outer()

    @property
    def _stat_c_xorpopcnt_outer(self):
        return self._core.statistics.collect_statistics_xorpopcnt_outer

    @property
    def _stat_get_xorpopcnt_pass_total(self):
        return self._core.statistics.get_stats_xorpopcnt_pass_total()

    @property
    def _stat_c_xorpopcnt_pass_total(self):
        return self._core.statistics.collect_statistics_xorpopcnt_pass_total

    @property
    def _stat_get_xorpopcnt_pass_inner(self):
        return self._core.statistics.get_stats_xorpopcnt_pass_inner()

    @property
    def _stat_c_xorpopcnt_pass_inner(self):
        return self._core.statistics.collect_statistics_xorpopcnt_pass_inner

    @property
    def _stat_get_xorpopcnt_pass_outer(self):
        return self._core.statistics.get_stats_xorpopcnt_pass_outer()

    @property
    def _stat_c_xorpopcnt_pass_outer(self):
        return self._core.statistics.collect_statistics_xorpopcnt_pass_outer

    @property
    def _stat_get_fullscprods_total(self):
        return self._core.statistics.get_stats_fullscprods_total()

    @property
    def _stat_c_fullscprods_total(self):
        return self._core.statistics.collect_statistics_fullscprods_total

    @property
    def _stat_get_fullscprods_inner(self):
        return self._core.statistics.get_stats_fullscprods_inner()

    @property
    def _stat_c_fullscprods_inner(self):
        return self._core.statistics.collect_statistics_fullscprods_inner

    @property
    def _stat_get_fullscprods_outer(self):
        return self._core.statistics.get_stats_fullscprods_outer()

    @property
    def _stat_c_fullscprods_outer(self):
        return self._core.statistics.collect_statistics_fullscprods_outer

    @property
    def _stat_get_filter_pass(self):
        return self._core.statistics.get_stats_filter_pass()

    @property
    def _stat_c_filter_pass(self):
        return self._core.statistics.collect_statistics_filter_pass

    @property
    def _stat_get_redsuccess_total(self):
        return self._core.statistics.get_stats_redsuccess_total()

    @property
    def _stat_c_redsuccess_total(self):
        return self._core.statistics.collect_statistics_redsuccess_total

    @property
    def _stat_get_2redsuccess_total(self):
        return self._core.statistics.get_stats_2redsuccess_total()

    @property
    def _stat_c_2redsuccess_total(self):
        return self._core.statistics.collect_statistics_2redsuccess_total

    @property
    def _stat_get_2redsuccess_inner(self):
        return self._core.statistics.get_stats_2redsuccess_inner()

    @property
    def _stat_c_2redsuccess_inner(self):
        return self._core.statistics.collect_statistics_2redsuccess_inner

    @property
    def _stat_get_2redsuccess_outer(self):
        return self._core.statistics.get_stats_2redsuccess_outer()

    @property
    def _stat_c_2redsuccess_outer(self):
        return self._core.statistics.collect_statistics_2redsuccess_outer

    @property
    def _stat_get_3redsuccess(self):
        return self._core.statistics.get_stats_3redsuccess()

    @property
    def _stat_c_3redsuccess(self):
        return self._core.statistics.collect_statistics_3redsuccess

    @property
    def _stat_get_dataraces_total(self):
        return self._core.statistics.get_stats_dataraces_total()

    @property
    def _stat_c_dataraces_total(self):
        return self._core.statistics.collect_statistics_dataraces_total

    @property
    def _stat_get_dataraces_2inner(self):
        return self._core.statistics.get_stats_dataraces_2inner()

    @property
    def _stat_c_dataraces_2inner(self):
        return self._core.statistics.collect_statistics_dataraces_2inner

    @property
    def _stat_get_dataraces_2outer(self):
        return self._core.statistics.get_stats_dataraces_2outer()

    @property
    def _stat_c_dataraces_2outer(self):
        return self._core.statistics.collect_statistics_dataraces_2outer

    @property
    def _stat_get_dataraces_3(self):
        return self._core.statistics.get_stats_dataraces_3()

    @property
    def _stat_c_dataraces_3(self):
        return self._core.statistics.collect_statistics_dataraces_3

    @property
    def _stat_get_dataraces_replaced_was_saturated(self):
        return self._core.statistics.get_stats_dataraces_replaced_was_saturated()

    @property
    def _stat_c_dataraces_replaced_was_saturated(self):
        return self._core.statistics.collect_statistics_dataraces_replaced_was_saturated

    @property
    def _stat_get_dataraces_sorting_blocked_cdb(self):
        return self._core.statistics.get_stats_dataraces_sorting_blocked_cdb()

    @property
    def _stat_c_dataraces_sorting_blocked_cdb(self):
        return self._core.statistics.collect_statistics_dataraces_sorting_blocked_cdb

    @property
    def _stat_get_dataraces_sorting_blocked_db(self):
        return self._core.statistics.get_stats_dataraces_sorting_blocked_db()

    @property
    def _stat_c_dataraces_sorting_blocked_db(self):
        return self._core.statistics.collect_statistics_dataraces_sorting_blocked_db

    @property
    def _stat_get_dataraces_get_p_blocked(self):
        return self._core.statistics.get_stats_dataraces_get_p_blocked()

    @property
    def _stat_c_dataraces_get_p_blocked(self):
        return self._core.statistics.collect_statistics_dataraces_get_p_blocked

    @property
    def _stat_get_dataraces_out_of_queue(self):
        return self._core.statistics.get_stats_dataraces_out_of_queue()

    @property
    def _stat_c_dataraces_out_of_queue(self):
        return self._core.statistics.collect_statistics_dataraces_out_of_queue

    @property
    def _stat_get_dataraces_insertions(self):
        return self._core.statistics.get_stats_dataraces_insertions()

    @property
    def _stat_c_dataraces_insertions(self):
        return self._core.statistics.collect_statistics_dataraces_insertions

    @property
    def _stat_get_collisions_total(self):
        return self._core.statistics.get_stats_collisions_total()

    @property
    def _stat_c_collisions_total(self):
        return self._core.statistics.collect_statistics_collisions_total

    @property
    def _stat_get_collisions_2inner(self):
        return self._core.statistics.get_stats_collisions_2inner()

    @property
    def _stat_c_collisions_2inner(self):
        return self._core.statistics.collect_statistics_collisions_2inner

    @property
    def _stat_get_collisions_2outer(self):
        return self._core.statistics.get_stats_collisions_2outer()

    @property
    def _stat_c_collisions_2outer(self):
        return self._core.statistics.collect_statistics_collisions_2outer

    @property
    def _stat_get_collisions_3(self):
        return self._core.statistics.get_stats_collisions_3()

    @property
    def _stat_c_collisions_3(self):
        return self._core.statistics.collect_statistics_collisions_3

    @property
    def _stat_get_collisions_nobucket(self):
        return self._core.statistics.get_stats_collisions_nobucket()

    @property
    def _stat_c_collisions_nobucket(self):
        return self._core.statistics.collect_statistics_collisions_nobucket

    @property
    def _stat_get_otflifts_total(self):
        return self._core.statistics.get_stats_otflifts_total()

    @property
    def _stat_c_otflifts_total(self):
        return self._core.statistics.collect_statistics_otflifts_total

    @property
    def _stat_get_otflifts_2inner(self):
        return self._core.statistics.get_stats_otflifts_2inner()

    @property
    def _stat_c_otflifts_2inner(self):
        return self._core.statistics.collect_statistics_otflifts_2inner

    @property
    def _stat_get_otflifts_2outer(self):
        return self._core.statistics.get_stats_otflifts_2outer()

    @property
    def _stat_c_otflifts_2outer(self):
        return self._core.statistics.collect_statistics_otflifts_2outer

    @property
    def _stat_get_otflifts_3(self):
        return self._core.statistics.get_stats_otflifts_3()

    @property
    def _stat_c_otflifts_3(self):
        return self._core.statistics.collect_statistics_otflifts_3

    @property
    def _stat_get_replacements_total(self):
        return self._core.statistics.get_stats_replacements_total()

    @property
    def _stat_c_replacements_total(self):
        return self._core.statistics.collect_statistics_replacements_total

    @property
    def _stat_get_replacements_list(self):
        return self._core.statistics.get_stats_replacements_list()

    @property
    def _stat_c_replacements_list(self):
        return self._core.statistics.collect_statistics_replacements_list

    @property
    def _stat_get_replacements_queue(self):
        return self._core.statistics.get_stats_replacements_queue()

    @property
    def _stat_c_replacements_queue(self):
        return self._core.statistics.collect_statistics_replacements_queue

    @property
    def _stat_get_replacements_large(self):
        return self._core.statistics.get_stats_replacements_large()

    @property
    def _stat_c_replacements_large(self):
        return self._core.statistics.collect_statistics_replacements_large

    @property
    def _stat_get_replacements_small(self):
        return self._core.statistics.get_stats_replacements_small()

    @property
    def _stat_c_replacements_small(self):
        return self._core.statistics.collect_statistics_replacements_small

    @property
    def _stat_get_replacementfailures_total(self):
        return self._core.statistics.get_stats_replacementfailures_total()

    @property
    def _stat_c_replacementfailures_total(self):
        return self._core.statistics.collect_statistics_replacementfailures_total

    @property
    def _stat_get_replacementfailures_queue(self):
        return self._core.statistics.get_stats_replacementfailures_queue()

    @property
    def _stat_c_replacementfailures_queue(self):
        return self._core.statistics.collect_statistics_replacementfailures_queue

    @property
    def _stat_get_replacementfailures_list(self):
        return self._core.statistics.get_stats_replacementfailures_list()

    @property
    def _stat_c_replacementfailures_list(self):
        return self._core.statistics.collect_statistics_replacementfailures_list

    @property
    def _stat_get_replacementfailures_prune(self):
        return self._core.statistics.get_stats_replacementfailures_prune()

    @property
    def _stat_c_replacementfailures_prune(self):
        return self._core.statistics.collect_statistics_replacementfailures_prune

    @property
    def _stat_get_sorting_total(self):
        return self._core.statistics.get_stats_sorting_total()

    @property
    def _stat_c_sorting_total(self):
        return self._core.statistics.collect_statistics_sorting_total

    @property
    def _stat_get_sorting_sieve(self):
        return self._core.statistics.get_stats_sorting_sieve()

    @property
    def _stat_c_sorting_sieve(self):
        return self._core.statistics.collect_statistics_sorting_sieve

    @property
    def _stat_get_buckets(self):
        return self._core.statistics.get_stats_buckets()

    @property
    def _stat_c_buckets(self):
        return self._core.statistics.collect_statistics_buckets

    @property
    def _stat_get_memory_buckets(self):
        return self._core.statistics.get_stats_memory_buckets()

    @property
    def _stat_c_memory_buckets(self):
        return self._core.statistics.collect_statistics_memory_buckets


    @property
    def _stat_get_memory_transactions(self):
        return self._core.statistics.get_stats_memory_transactions()

    @property
    def _stat_c_memory_transactions(self):
        return self._core.statistics.collect_statistics_memory_transactions

    @property
    def _stat_get_memory_snapshots(self):
        return self._core.statistics.get_stats_memory_snapshots()

    @property
    def _stat_c_memory_snapshots(self):
        return self._core.statistics.collect_statistics_memory_snapshots


    # This dictionary controls how statistics are exported / displayed.
    #
    # Format is as follows: key equals the C++ = decl.pxd = _stat_get_ name
    # Value is [SequenceID, short description, long description, algs, OPTIONAL: repr]
    # where  SequenceID is a number used to determine in which order we write output
    #        short description is the prefix used in (short) humand-readable output
    #        long description is a meaningful "docstring"
    #        algs is a set of algorithms where this statistic is meaningful
    #        repr is optional and is passed to the Accumulator inside the TreeTracer as its
    #           repr argument. Set to "max" to output the max value instead of the sum.

    # Note: Buckets == Filtered lists (the terminonlogy depends on the algorithm)
    all_statistics = {
        "reds_total"            : [10,  "R   :",  "total number of reduction attempts",                                 {"bgj1", "triple_mt"}],
        "2reds_total"           : [11,  "R2  :",  "total number of 2-reduction attempts",                               {        "triple_mt"}],
        "2reds_outer"           : [12,  "R2-o:",  "number of 2-reduction attempts in filtering / bucketing phase",      {"bgj1", "triple_mt"}],
        "2reds_inner"           : [13,  "R2-i:",  "number of 2-reduction attempts in sieving phase",                    {"bgj1", "triple_mt"}],
        "3reds"                 : [14,  "R3  :",  "number of 3-reduction attempts",                                     {        "triple_mt"}],
        "xorpopcnt_total"       : [20,  "XPC :",  "total number of xor-popcounts inside sieve",                         {"bgj1", "triple_mt"}],
        "xorpopcnt_outer"       : [21,  "XPCo:",  "xor-popcounts while filtering / bucketing",                          {"bgj1", "triple_mt"}],
        "xorpopcnt_inner"       : [22,  "XPCi:",  "xor-popcounts while sieving within bucket",                          {"bgj1", "triple_mt"}],
        "xorpopcnt_pass_total"  : [30,  "XPS :",  "total successful xor-popcounts inside sieve",                        {"bgj1", "triple_mt"}],
        "xorpopcnt_pass_outer"  : [31,  "XPSo:",  "successful xor-popcounts while bucketing",                           {"bgj1", "triple_mt"}],
        "xorpopcnt_pass_inner"  : [32,  "XPSi:",  "successful xor-popcounts within bucket",                             {"bgj1", "triple_mt"}],
        "fullscprods_total"     : [40,  "ScP :",  "total full scalar products in sieve (w/o Entry recomputations)",     {"bgj1", "triple_mt"}],
        "fullscprods_outer"     : [41,  "ScPo:",  "full scalar products while bucketing",                               {"bgj1", "triple_mt"}],
        "fullscprods_inner"     : [42,  "ScPi:",  "full scalar products while working inside bucket",                   {"bgj1", "triple_mt"}],
        "filter_pass"           : [50,  "PASS:",  "total number of points that end up in a bucket / filtered list",     {"bgj1", "triple_mt"}],
        "buckets"               : [55,  "BUCK:",  "total number of buckets / filtered lists",                           {"bgj1", "triple_mt"}],
        "redsuccess_total"      : [60,  "Suc :",  "total number of reduction successes (== put into transaction-db)",   {"bgj1", "triple_mt"}],
        "2redsuccess_total"     : [61,  "Suc2:",  "total number of 2-reductions successes",                             {        "triple_mt"}],
        "2redsuccess_outer"     : [62,  "Suco:",  "2-reduction successes in bucketing phase",                           {"bgj1", "triple_mt"}],
        "2redsuccess_inner"     : [63,  "Suci:",  "2-reduction successes in sieving phase",                             {"bgj1", "triple_mt"}],
        "3redsuccess"           : [64,  "Suc3:",  "3-reduction successes",                                              {        "triple_mt"}],
        "dataraces_total"       : [200, "ERR :",  "Total errors (mostly data races)",                                   {"bgj1", "triple_mt"}],
        "dataraces_2outer"      : [201, "ERRo:",  "Data races corrupting (c)db reads while bucketing",                  {"bgj1", "triple_mt"}],
        "dataraces_2inner"      : [202, "ERRi:",  "Data races corrupting (c)db reads while sieving",                    {"bgj1", "triple_mt"}],
        "dataraces_3"           : [203, "ERR3:",  "Data races corrupting (c)db reads in 3-reductions",                  {        "triple_mt"}],
        "dataraces_replaced_was_saturated" : [204, "ERRX:", "Overwriting vector below saturation bound",                {"bgj1", "triple_mt"}], #Note : This might affect detection of saturation condition and should be considered a bug!
        "dataraces_sorting_blocked_cdb" : [205, "ERRC:", "Sorting has to wait for cdb writes by other threads",         {        "triple_mt"}],
        "dataraces_sorting_blocked_db"  : [206, "ERRD:", "Sorting has to wait for db writes by other threads",          {        "triple_mt"}],
        "dataraces_get_p_blocked" :       [207, "ERRB:", "Thread job start got blocked by mutex",                       {        "triple_mt"}],
        "dataraces_out_of_queue" :        [208, "ERRQ:", "queue has been exhausted",                                    {        "triple_mt"}],
        "dataraces_insertions"  :         [209, "ERRI:", "Concurrent insertions reservation failures",                  {        "triple_mt"}],
        "collisions_total"      : [70,  "C   :",  "Total number of Collisions while sieving",                           {"bgj1", "triple_mt"}],
        "collisions_2outer"     : [71,  "C2-o:",  "Collisions while bucketing",                                         {"bgj1", "triple_mt"}], #Not sure about bgj1
        "collisions_2inner"     : [72,  "C2-i:",  "Collisions while working inside bucket",                             {"bgj1", "triple_mt"}],
        "collisions_3"          : [73,  "C3  :",  "Collisions coming from 3-reductions",                                {        "triple_mt"}],
        "collisions_nobucket"   : [74,  "C-xB:",  "Collisions preventing an element from being put in bucket",          {        "triple_mt"}],
        "otflifts_total"        : [80,  "OTF :",  "Total number of OTFLift attempts while sieving",                     {"bgj1", "triple_mt"}],
        "otflifts_2outer"       : [81,  "OTFo:",  "OTFLift attempts while bucketing",                                   {"bgj1", "triple_mt"}],
        "otflifts_2inner"       : [82,  "OTFi:",  "OTFLift attempts while working inside bucket",                       {"bgj1", "triple_mt"}],
        "otflifts_3"            : [83,  "OTF3:",  "OFTLift attempts involving 3 vectors",                               {        "triple_mt"}],
        "replacements_total"    : [90,  "Rep :",  "Total number of successful DB replacements during sieve",            {"bgj1", "triple_mt"}],
        "replacements_list"     : [91,  "RepL:",  "Total number of successful DB replacements into list part",          {        "triple_mt"}], #Note: For algorithms that do not distinguish, all replacements are into list part
        "replacements_queue"    : [92,  "RepQ:",  "Total number of successful DB replacements into queue part",         {        "triple_mt"}],
        "replacements_large"    : [93,  "Rep+:",  "Number of block insertion attemts of large blocks",                  {        "triple_mt"}], #TODO: Add these for other algs
        "replacements_small"    : [94,  "Rep-:",  "Number of block insertion attemts of small blocks",                  {        "triple_mt"}], #TODO: Add these for other algs
        "replacementfailures_total" : [100, "RpF :", "Total number of DB replacement failures",                         {"bgj1", "triple_mt"}],
        "replacementfailures_list"  : [101, "RpFL:", "Number of DB replacement failures into list (vector too long)",   {        "triple_mt"}], #Note: For bgj1, ignore the list part. This simply counts when a vector would overwrite one that was shorter.
        "replacementfailures_queue" : [102, "RpFQ:", "Number of DB replacement failures into queue (vector too long)",  {        "triple_mt"}],
        "replacementfailures_prune"  : [103, "RpFP:", "Number of DB replacement failures (Pruned)",                     {        "triple_mt"}],
        "sorting_total"         : [150, "Sort:", "Total number of sortings during sieve",                               {"bgj1", "triple_mt"}],
        "sorting_sieve"         : [151, "Sort:", "Total number of sortings during sieve",                               {}                   ], #Note: The distinction is only here if we later add other sorting events to also be counted.
        "memory_buckets"        : [160, "MemB:", "Total number of bucket elements we reserved memory for",              {        "triple_mt"}, "max"],
        "memory_transactions"   : [161, "MemT:", "Total number of transactions we reserved memory for",                 {        "triple_mt"}, "max"],
        "memory_snapshots"      : [162, "MemS:", "Maximum number of concurrent CDB snapshots",                          {        "triple_mt"}, "max"], # Note: For bgj1, this value is always 2 (and hence not collected). For other sieves, it is 1.
    }

    @property
    def stats(self):
        "Returns all collected statistics of the current sieve as a dictionary"
        ret = {"cdb-size:" : self._core.cdb.size()}
        for key in Siever.all_statistics:
            if(getattr(self,"_stat_c_" + key) == True):
                ret[key] = getattr(self, "_stat_get_" + key)
        return ret

    @classmethod
    def crop_stats(cls, stats, alg):
        """
        Takes a dictionary stats (presumbably from stats) and returns a (shallow entrywise) copy of
        it with only the statistics relevant for alg.

        :param stats: statistics dictionary
        :param alg: algorithm

        """

        ret = dict(stats) #need to make a copy. Note that the entries of dict are copied shallowly (if I understand Python correctly)
        if alg == "all":
            return ret
        for key in stats:
            if key in cls.all_statistics:
                if not alg in cls.all_statistics[key][3]:
                    del ret[key]
        return ret

    def stats_for_alg(self, alg):
        """
        Returns a dictionary of collected statistics that are meaningful for algorithm alg.
        """
        return self.crop_stats(self.stats, alg)

    def get_print_stats(self, alg="all", detail = "short"):
        """
        Gets list of collected statistics meaningful for algorithm ``alg``.

        :param alg: sieving algorithm
        :param detail: One of "short", "long", "key"

        """
        # this sorts the dictionary according to the sequenceIDs (and after that lexicographically)
        # in Siever.all_statistics (with additional items like "cdb-size" having sequenceID 0)
        l = sorted(self.stats_for_alg(alg).items(), key=lambda kv: Siever.all_statistics.get(kv[0], 0) )
        if detail == "short":
            l = [(Siever.all_statistics.get(k,[None,k])[1],v) for (k,v) in l]
            return l
        elif detail == "long":
            l = [(Siever.all_statistics.get(k,[None,None,k])[2],v) for (k,v) in l]
            return l
        elif detail == "key":
            return l
        raise ValueError("detail must be one of short, long or key")

    def get_stat(self, key):
        """
        Get statistic named by key if it is actually collected (this depends on compile-time options
        for the C++ part).  If support was not compiled in, returns None.
        """
        if getattr(self,"_stat_c_" + key) == True:
            return getattr(self,"_stat_get_" + key)
        return None

    ## End of new statistics ##


    # For debugging purposes
    # def print_histo(self):
    #     histo = self.db_stats()
    #     for (r, c) in histo[30:40:2]:
    #         print "%.2f:%.2f "%(r,c),
    #     print

    def resize_db(self, N, large=0):
        """
        Resize db to ``N``.

        :param N: new size
        :param large: If large > 0, will sample large fresh vectors. Otherwise
        sample by combination of current vectors

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> from g6k import Siever
            >>> siever= Siever(IntegerMatrix.random(50, "qary", k=25, bits=10), seed=0x1337)
            >>> siever.initialize_local(0, 1, 11)
            >>> siever()
            >>> len(siever)
            20
            >>> S = set(siever.itervalues())
            >>> siever.db_size()
            20
            >>> siever.resize_db(len(siever)/2)

            >>> len(siever)
            10

        """

        if N < self.db_size():
            self.shrink_db(N)
        elif N > self.db_size():
            self.grow_db(N, large)

    def shrink_db(self, N):
        """
        Shrinks db to size (at most) N. This preferentially deletes vectors that are longer.
        """
        if N>0:
            assert(self.initialized)
        self._core.shrink_db(int(ceil(N)))

    def grow_db(self, N, large=0):
        assert(self.initialized)
        self._core.grow_db(<unsigned long>(ceil(N)), <int>(large))

    def histo_index(self, r):
        """
        Round a length normalized by gh to an index for the histogram statistics.

        :param r: squared length / gh.

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.histo_index(1.551)
            166
            >>> g6k.histo_index(4./3.)
            100
            >>> g6k.histo_index(.98)
            0
            >>> g6k.histo_index(2.2)
            299

        """
        return self._core.histo_index(r)

    def check_saturation(self, min_n=50):
        """
        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 10, 30)
            >>> g6k()
            >>> g6k.check_saturation(10)

        """
        if self.n < min_n:
            return

        histo = self.db_stats()
        i = self.histo_index(self.params.saturation_radius)
        sat = max(histo[i:])

        if sat < .99 * self.params.saturation_ratio:
            message = "saturation %.3f at [b:%d P: %d - %d / %d], radius %.3f, goal: %.3f"
            message = message%(sat, self.n, self.l, self.r, self.full_n, self.params.saturation_radius, self.params.saturation_ratio)
            logging.warning(message)
            logging.info("Could not reach saturation, progress may slow.")
            raise SaturationError()


    def bgj1_sieve(self, reset_stats=True):
        assert(self.initialized)
        if self.n < 40:
            logging.warning("bgj1_sieve not recommended below dimension 40")

        if reset_stats:
            self.reset_stats()

        N = self.params.db_size_factor * self.params.db_size_base ** self.n
        self.resize_db(N)
        B = self.params.bgj1_bucket_size_factor * (N ** self.params.bgj1_bucket_size_expo)

        y = B/N
        x = betaincinv((self.n+1.)/2., .5, y)
        alpha = (1 - x)**.5

        sig_on()
        self._core.bgj1_sieve(alpha)
        sig_off()

        self.check_saturation()
        return self.stats

    def bdgl_sieve(self, blocks=None, buckets=None, reset_stats=True, check_saturation=True):
        assert(self.initialized)
        if self.n < 40:
            logging.warning("bdgl_sieve not recommended below dimension 40")


        if reset_stats:
            self.reset_stats()

        N = self.params.db_size_factor * self.params.db_size_base ** self.n
        self.resize_db(N)

        if blocks is None:
            blocks =  self.params.bdgl_blocks

        if blocks not in [1,2,3]:
            logging.warning("bdgl_sieve only supports 1, 2, or 3 blocks")

        blocks = min(3, max(1, blocks))
        blocks = min(int(self.n / 28), blocks)

        if buckets is None:
            buckets = self.params.bdgl_bucket_size_factor * 2.**((blocks-1.)/(blocks+1.)) * self.params.bdgl_multi_hash**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))

        buckets = min(buckets, self.params.bdgl_multi_hash * N / self.params.bdgl_min_bucket_size)
        buckets = max(buckets, 2**(blocks-1))

        sig_on()
        self._core.bdgl_sieve(buckets, blocks, self.params.bdgl_multi_hash)
        sig_off()

        if check_saturation:
            self.check_saturation()

    def nv_sieve(self, reset_stats=True):
        assert(self.initialized)
        if self.n < 40:
            logging.warning("nv sieve not recommended below dimension 40")

        if reset_stats:
            self.reset_stats()

        N = self.params.db_size_factor * self.params.db_size_base ** self.n
        self.resize_db(N)

        sig_on()
        self._core.nv_sieve()
        sig_off()

        self.check_saturation()



    def hk3_sieve(self, size_t max_db_size = 0, reset_stats=True):
        assert(self.initialized)
        if self.n < 40:
            logging.warning("triple_mt sieve not recommended below dimension 40")
            # Actually, that recommendation threshold comes from bgj1 and there is not much reason
            # to apply it to other algorithms. Since silently replacing the algorithm is making
            # people who debug stuff upset, I changed it to a warning. return
            # self.gauss_sieve(reset_stats=reset_stats)

        if reset_stats:
            self.reset_stats()

        # print self.params.db_size_base

        if max_db_size == 0:
            max_db_size = self.params.db_size_factor * self.params.db_size_base ** self.n

        self.resize_db(max_db_size)
        B = self.params.bgj1_bucket_size_factor * (max_db_size ** self.params.bgj1_bucket_size_expo)

        y = B/max_db_size
        x = betaincinv((self.n+1.)/2., .5, y)
        alpha = (1 - x)**.5


        sig_on()
        self._core.hk3_sieve(alpha)
        sig_off()

        self.check_saturation()
        return self.stats

    def gauss_sieve(self, size_t max_db_size=0, reset_stats=True):

        assert(self.initialized)
        if reset_stats:
            self.reset_stats()

        if max_db_size==0:
            max_db_size = 500 + 10*self.n + 2 * self.params.db_size_factor * self.params.db_size_base ** self.n

        if self.db_size() > max_db_size:
            self.resize_db(max_db_size)

        sig_on()
        self._core.gauss_sieve(max_db_size)
        sig_off()

        self.check_saturation()
        return self.stats


    def __call__(self, alg=None, reset_stats=True, tracer=dummy_tracer):
        assert(self.initialized)

        # Check choice of sieve algorithm preemptively, to avoid incorrect user
        # choices  being overwritten by default or crossover leading to non-deterministic
        # raise of the error

        valid_sieves = ["nv", "bgj1", "gauss", "hk3", "bdgl", "bdgl1", "bdgl2", "bdgl3"]
        if alg is not None and alg not in valid_sieves:
            raise NotImplementedError("Sieve Algorithm '%s' invalid. "%(alg) + "Please choose among "+str(valid_sieves) )

        if self.params.default_sieve not in valid_sieves:
            raise NotImplementedError("Sieve Algorithm '%s' invalid. "%(self.params.default_sieve) + "Please choose among "+str(valid_sieves) )

        if alg is None:
            if self.n < self.params.gauss_crossover:
                alg = "gauss"
            else:
                alg = self.params.default_sieve

        # TODO: Make the context string identical to the name taken as a command-line parameter,
        # once we drop some sieve algorithms

        if alg == "nv":
            with tracer.context("nv"):
                self.nv_sieve(reset_stats=reset_stats)
        elif alg == "bgj1":
            with tracer.context("bgj1"):
                self.bgj1_sieve(reset_stats=reset_stats)
        elif alg == "bdgl":
            with tracer.context("bdgl"):
                self.bdgl_sieve(reset_stats=reset_stats)
        elif alg == "bdgl1":
            with tracer.context("bdgl"):
                self.bdgl_sieve(blocks=1, reset_stats=reset_stats)
        elif alg == "bdgl2":
            with tracer.context("bdgl"):
                self.bdgl_sieve(blocks=2, reset_stats=reset_stats)
        elif alg == "bdgl3":
            with tracer.context("bdgl"):
                self.bdgl_sieve(blocks=3, reset_stats=reset_stats)
        elif alg == "gauss":
            with tracer.context("gauss"):
                self.gauss_sieve(reset_stats=reset_stats)
        elif alg == "hk3": #Multi-threaded 3Sieve, keep both for now
            with tracer.context("hk3"):
                self.hk3_sieve(reset_stats=reset_stats)
        else:
            # The algorithm should have been preemptively checked
            assert(False)


    def extend_left(self, offset=1):
        """
        Extend the context to the left:

            - change the context

            - pad all vectors in db/cdb with 0 on the left

            - refresh the database with parameters ``babai_index=l_index=offset``

        We illustrate the behavior of this function with a small example.  We start by setting up
        our instance.  We also keep a copy of the last 16 vectors in the database for comparison::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 1, 5)
            >>> g6k()
            >>> db = list(g6k.itervalues())[:16];
            >>> abs(sum(db[0])) == 1
            True

        This function, changes the context, i.e. the left index is reduced::

            >>> g6k.l, g6k.r
            (1, 5)

            >>> g6k.extend_left(); g6k.l, g6k.r
            (0, 5)

        """
        # TODO the documentation needs fixing
        assert(self.initialized)
        sig_on()
        self._core.extend_left(offset)
        sig_off()

    def shrink_left(self, lp=1):
        """
        Shrink the context to the left (threaded)

            - change the context

            - shrink all vectors in db/cdb on the left

            - refresh the database with ``(babai_index=l_index=0)``

        We illustrate the behavior of this function with a small example.  We start by setting up
        our instance.  We also keep a copy of the last 16 vectors in the database for comparison::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 1, 10)
            >>> g6k()
            >>> db = list(g6k.itervalues())[:16]
            >>> out = db[0]
            >>> out if out[0] > 0 else tuple([-x for x in out])
            (1, 0, 0, 0, 0, 0, 0, 0, 0)

        This function, changes the context, i.e. the left index is increased::

            >>> g6k.l, g6k.r
            (1, 10)

            >>> g6k.shrink_left(); g6k.l, g6k.r
            (2, 10)

        """
        # TODO the documentation needs fixing
        assert(self.initialized)
        sig_on()
        self._core.shrink_left(lp)
        sig_off()

    def extend_right(self, offset=1):
        """
        Extend the context to the right:

            - change the context

            - extend all vectors in db/cdb on the right with zero

            - refresh the database with parameters ``(babai_index=l_index=0)``

        We illustrate the behavior of this function with a small example.  We start by setting up
        our instance::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=15, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 1, 5)
            >>> g6k.l, g6k.r
            (1, 5)
            >>> g6k()

        We keep a copy of the database for comparison::

            >>> db0 = list(g6k.itervalues())

        This function, changes the context, i.e. the right index is extended::

            >>> g6k.extend_right(); g6k.l, g6k.r
            (1, 6)

        All vectors are extended with zero on the right::

            >>> db1 = list(g6k.itervalues())
            >>> len(db1) == len(db0)
            True
            >>> set(db1).difference([tuple(list(v) + [0]) for v in db0]) in (set(), set([]))
            True

        """
        assert(self.initialized)
        # in dual mode this is really an extend left (under the hood) so no need to update gso
        if not self.params.dual_mode:
            self.update_gso(self.ll, self.r + offset)
        sig_on()
        self._core.extend_right(offset)
        sig_off()

    def insert(self, kappa, v):
        """
        Insert a vector in the GSO basis, and update the siever accordingly (l++, n--, r fixed, db is updated)

        :param kappa: position at which to insert improved vector
        :param v: Improved vector expressed in base B[0 … r-1]
        """
        assert(self.initialized)
        assert(len(v) == self.r)
        m = self.full_n

        full_j = where(abs(v) == 1)[0][-1]

        if full_j < self.l:
            print full_j, self.l
            print v
            raise NotImplementedError('Can only handle vectors with +/- 1 in sieving context (have you deactivated param.unitary_only ?)')

        assert kappa <= self.l

        if v[full_j] == -1:
            v *= -1

        self.M.UinvT.gen_identity()
        self.M.U.gen_identity()

        if not self.params.dual_mode:
            with self.M.row_ops(kappa, self.r):
                for i in range(kappa, self.r):
                    if i != full_j:
                        self.M.row_addmul(full_j, i, v[i])
                self.M.move_row(full_j, kappa)
        else:
            with self.M.row_ops(m-self.r, m-kappa):
                # perform the dual operations to insert in the dual
                for i in range(kappa, self.r):
                    if i != full_j:
                        self.M.row_addmul(m-1-i, m-1-full_j, -v[i])
                self.M.move_row(m-1-full_j, m-1-kappa)


        new_l = self.l + 1
        new_n = self.n - 1

        self.split_lll(self.ll, new_l, self.r)

        cdef np.ndarray T = zeros((new_n, self.n), dtype=int64, order='C')

        if not self.params.dual_mode:
            for i in range(new_n):
                for j in range(self.n):
                    T[i][j] = self.M.UinvT[new_l + i][self.l + j]
        else:
            for i in range(new_n):
                for j in range(self.n):
                    T[i][j] = self.M.U[m-1-(new_l + i)][m-1-(self.l + j)]

        # update the basis (GSO or integral) of the lattice after insert
        sig_on()
        self._core.gso_update_postprocessing(new_l, self._core.r, <long*>T.data)
        sig_off()


    def lll(self, l, r):
        """
        Run LLL from l to r. (l and r are automatically reflected in dual mode)

        :param l: left index
        :param r: right index
        :returns: transposed inverse of transformation matrix

        ..  note:: This invalidates the sieve-contexts. g6k.db should be reset after doing this.

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(40, "qary", k=20, bits=20))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 10, 30)
            >>> _ = g6k.lll(10, 30)

        """

        lll = LLL.Reduction(self.M)
        if not self.params.dual_mode:
            lll(l, l, r)
        else:
            m = self.full_n
            lll(m-r, m-r, m-l)

        self.initialized=False


    def split_lll(self, lp, l, r):
        """
        Run partials LLL first between lp and l and then between l and r. text
        does not change.

        :param lp: left index
        :param l: middle index
        :param r: right index

        ..  note:: This enforces that the projected sublattice between l and r does not change
            and thus the sieving can be maintained. This maintaince is /not/ done here. In dual mode
            this requires to limit the size reduction in one of the calls so the result might not be
            fully size reduced.

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(40, "qary", k=20, bits=20))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 10, 30)
            >>> _ = g6k.split_lll(0, 10, 30)

        """

        lll_ = LLL.Reduction(self.M)
        m = self.full_n
        if not self.params.dual_mode:
            lll_(lp, lp, l)
            lll_(l, l, r)
        else:
            lll_(m-r, m-r, m-l)
            lll_(m-l, m-l, m-lp)

        self.update_gso(self.ll, self.r)

    def best_lifts(self):
        """
        Output the best lift (conditioned on improving that Gram-Schmidt vector) at each position i
        from 0 to l (INCLUDED) as a list of triples (i, length, vector).  If param.otf_lift=False,
        this triggers computation of the lift of the current database.  If param.otf_lift=True,
        retrieve best_lift_so_far which is continuously maintained considering all the vectors
        visited during Sieve (and even some more if oversieve_radius > 0).

        EXAMPLES::

            >>> from fpylll import IntegerMatrix, LLL, FPLLL
            >>> FPLLL.set_random_seed(0x1337)
            >>> from g6k import Siever
            >>> A = LLL.reduction(IntegerMatrix.random(30, "qary", k=25, bits=10))
            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 0, 20)
            >>> g6k.best_lifts()
            []

            >>> g6k()
            >>> bl = g6k.best_lifts()
            >>> id, nrm, w = bl[0]
            >>> id, int(nrm)
            (0, 197445)
            >>> sum([v**2 for v in A.multiply_left(w)])
            197445

            >>> g6k = Siever(A)
            >>> g6k.initialize_local(0, 10, 30)
            >>> g6k.best_lifts()
            []

            >>> g6k()
            >>> bl = g6k.best_lifts()
            >>> id, nrm, w = bl[0]
            >>> id, round(nrm)
            (0, 194629)
            >>> sum([v**2 for v in A.multiply_left(w)])
            194629

        """
        assert(self.initialized)

        cdef np.ndarray vecs = zeros((self.l+1, self.r), dtype=int64)
        cdef np.ndarray lens = zeros((self.l+1), dtype=float64)
        self._core.best_lifts(<long *>vecs.data, <double*>lens.data)  # // , unitary_only)
        L = []
        for i in range(self.l+1):
            if lens[i] > 0.:
                L.append((i, lens[i], vecs[i]))
        return L


    def insert_best_lift(self, scoring=(lambda index, nlen, olen, aux: True), aux=None):
        """
        Consider all best lifts, score them, and insert the one with the best score.

            - The scoring function takes three variable inputs ``index, new_length, old_length`` as
              parameters, where ``index, new_length`` are from ``best_lifts``, and ``old_length`` is
              the current gram-schmidt length at ``index`

            - it also takes a fixed auxilliary input

            - A score of ``None`` or ``False`` is ignored.  If all scores are ``None``, nothing is
              inserted.

            - Otherwise, insert at the position optimizing the score.  In case of ties, the smallest
              index ``index`` is chosen.

            - Returns the position of insertion (if any).

        EXAMPLE::

            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(80, "qary", k=40, bits=20)
            >>> A = LLL.reduction(A)
            >>> sieve = Siever(A)
            >>> sieve.initialize_local(10, 20, 50)
            >>> sieve()
            >>> _ = sieve.insert_best_lift()

        """
        assert(self.initialized)

        L = self.best_lifts()
        if len(L) == 0:
            return None

        score_list = [(scoring(index, nlen, self.M.get_r(index, index), aux), -index, v) for (index, nlen, v) in L]
        score_list = [(a, b, c) for (a,b,c) in score_list if a]

        # print [("%.3f"%a, b) for (a,b,c) in score_list]
        # print
        if not score_list:
            (best_score, best_i, best_v) = (None, None, None)
        else:
            (best_score, best_i, best_v) = max(score_list)

        if best_score is None or not best_score:
            return None

        self.insert(-best_i, best_v)
        return -best_i


    def show_cpu_stats(self):
        # TODO this uses C++'s cout, which we shouldn't use. For example, this won't show up in a
        # Jupyter notebook.
        show_cpu_stats()

    def db_stats(self, absolute_histo=False):
        # TODO find a better return type

        if not len(self):
            raise ValueError("Database is empty.")

        cdef np.ndarray tmp_histo = zeros(self._core.size_of_histo, dtype=int64)

        self._core.db_stats(<long*>tmp_histo.data)

        if absolute_histo:
            return [(tmp_histo[i]) for i in range(self._core.size_of_histo)]
        else:
            return [(2 * tmp_histo[i] / (1+i*(1./self._core.size_of_histo))**(self.n/2.)) for i in range(self._core.size_of_histo)]


# For backward compatibility with old pickles
from siever_params import unpickle_params
