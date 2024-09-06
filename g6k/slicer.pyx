#cython: linetrace=True

from libcpp.vector cimport vector
from numpy import zeros, float32, float64, int64, matrix, array, where, matmul, identity, dot
cimport numpy as np


cdef class Randomized_slicer(object):

    def __init__(self, seed = 0):
        self._core = new Randomized_slicer_c(<unsigned long>seed)

    def grow_db_with_target(self, target, n_per_target):
        assert(self.initialized)
        cdef np.ndarray target_f = zeros(len(target), dtype=float64)
        for i in range(len(target)):
            target_f[i] = target[i]
        self._core.grow_db_with_target(<float*>target_f.data, n_per_target)