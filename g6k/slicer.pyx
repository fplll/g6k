#cython: linetrace=True
import numpy as np
from libcpp.vector cimport vector
from numpy import zeros, float32, float64, int64, matrix, array, where, matmul, identity, dot
from cysignals.signals cimport sig_on, sig_off
cimport numpy as np

#from siever import Siever
from cython.operator import dereference
from decl cimport MAX_SIEVING_DIM
from decl cimport CompressedEntry, Entry, Entry_t

cdef class RandomizedSlicer(object):

    def __init__(self, Siever sieve, seed = 0):
        self._core = new RandomizedSlicer_c(dereference(sieve._core), <unsigned long>seed)

    def grow_db_with_target(self, target, size_t n_per_target):
        cdef np.ndarray target_f = zeros(MAX_SIEVING_DIM, dtype=np.float64)
        #np.ndarray[np.double_t,ndim=1] target_f

        for i in range(len(target)):
            target_f[i] = target[i]
        sig_on()
        #print("target_f:", target_f)
        self._core.grow_db_with_target(<double*> target_f.data, n_per_target)
        sig_off()

    def bdgl_like_sieve(self, size_t nr_buckets, size_t blocks, size_t multi_hash, len_bound):
        sig_on()
        self._core.bdgl_like_sieve(nr_buckets, blocks, multi_hash, len_bound)
        sig_off()

    def itervalues_t(self):
        """
        Iterate over all entries in the target database (in the order determined by the compressed database)

        """
        cdef Entry_t *e;

        for i in range(self._core.cdb_t.size()):
            e = &self._core.db_t[self._core.cdb_t[i].i]
            r = [e.yr[j] for j in range(self._core.n)]
            yield tuple(r)
