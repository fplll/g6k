from g6k.decl cimport SieverParams as SieverParams_c

cdef class SieverParams(object):
    cdef SieverParams_c _core
    cpdef _set(self, str key, object value)
    cpdef object _get(self, str key)
    cdef int _read_only
    cdef tuple _cppattr
    cdef dict _pyattr
