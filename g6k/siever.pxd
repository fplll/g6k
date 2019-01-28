# -*- coding: utf-8 -*-
"""
"""

from g6k.decl cimport Siever as Siever_c
from g6k.siever_params cimport SieverParams

cdef class Siever(object):
    cdef Siever_c *_core
    cdef public object M
    cdef SieverParams _params
    cdef object initialized
