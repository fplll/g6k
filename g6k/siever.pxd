# -*- coding: utf-8 -*-
"""
"""

from g6k.decl cimport Siever as Siever_c
from g6k.siever_params cimport SieverParams

from libc.stdint cimport int16_t, int32_t, uint64_t

cdef extern from "../kernel/siever.h" nogil:
    cdef const int  MAX_SIEVING_DIM

    ctypedef double FT
    ctypedef float  LFT
    ctypedef int16_t ZT
    ctypedef int32_t IT
    ctypedef uint64_t UidType

cdef class Siever(object):
    cdef Siever_c *_core
    cdef public object M
    cdef SieverParams _params
    cdef object initialized



