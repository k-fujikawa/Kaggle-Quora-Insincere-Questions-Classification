# cython: language_level=3
import numpy as np
cimport numpy as np
from multiprocessing import Pool


cdef class ApplyNdArray:
    cdef func
    cdef int processes

    def __init__(self, func, processes=1):
        self.func = func
        self.processes = processes

    def __call__(self, np.ndarray[unicode] arr):
        if self.processes == 1:
            return self.apply(arr)
        else:
            return self.apply_parallel(arr)

    cpdef np.ndarray[object] apply(self, np.ndarray[unicode] arr):
        cdef int i
        cdef int n = len(arr)
        cdef np.ndarray[object] res = np.empty(n, dtype=object)
        for i in range(n):
            res[i] = self.func(arr[i])
        return res

    cpdef np.ndarray[object] apply_parallel(self, np.ndarray[unicode] arr):
        cdef list arrs = np.array_split(arr, self.processes)
        with Pool(processes=self.processes) as pool:
            outputs = pool.map(self.apply, arrs)
        return np.concatenate(outputs, axis=0)
