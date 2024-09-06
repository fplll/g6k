# -*- coding: utf-8 -*-
"""
"""

from g6k.decl cimport Randomized_slicer as Randomized_slicer_c
#from g6k.siever_params cimport SieverParams
#from g6k.siever cimport Siever

from libc.stdint cimport int16_t, int32_t, uint64_t

cdef extern from "../kernel/siever.h" nogil:
        cdef const int  MAX_SIEVING_DIM

        ctypedef double FT
        ctypedef float  LFT
        ctypedef int16_t ZT
        ctypedef int32_t IT
        ctypedef uint64_t UidType

cdef class Randomized_slicer(object):
    cdef Randomized_slicer_c *_core
    cdef object initialized