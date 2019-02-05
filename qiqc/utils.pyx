# cython: language_level=3
import numpy as np
cimport numpy as np
from multiprocessing import Pool


cdef class ApplyNdArray:
    cdef func
    cdef dtype
    cdef dims
    cdef int processes

    def __init__(self, func, processes=1, dtype=object, dims=None):
        self.func = func
        self.processes = processes
        self.dtype = dtype
        self.dims = dims

    def __call__(self, arr):
        if self.processes == 1:
            return self.apply(arr)
        else:
            return self.apply_parallel(arr)

    cpdef apply(self, arr):
        cdef int i
        cdef int n = len(arr)
        if self.dims is not None:
            shape = (n, *self.dims)
        else:
            shape = n
        cdef res = np.empty(shape, dtype=self.dtype)
        for i in range(n):
            res[i] = self.func(arr[i])
        return res

    cpdef apply_parallel(self, arr):
        cdef list arrs = np.array_split(arr, self.processes)
        with Pool(processes=self.processes) as pool:
            outputs = pool.map(self.apply, arrs)
        return np.concatenate(outputs, axis=0)
