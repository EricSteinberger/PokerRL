# Copyright (c) 2019 Eric Steinberger


import ctypes
import os

import numpy as np


class CppWrapper:
    """
    Baseclass to wrap a C++ library using ctypes
    """
    ARR_2D_ARG_TYPE = np.ctypeslib.ndpointer(dtype=np.intp, ndim=1, flags='C')
    CPP_LIB_FILE_ENDING = "dll" if os.name == 'nt' else "so"

    def __init__(self, path_to_dll):
        self._clib = ctypes.cdll.LoadLibrary(path_to_dll)

    @staticmethod
    def np_1d_arr_to_c(np_arr):
        return ctypes.c_void_p(np_arr.ctypes.data)

    @staticmethod
    def np_2d_arr_to_c(np_2d_arr):
        return (np_2d_arr.__array_interface__['data'][0]
                + np.arange(np_2d_arr.shape[0]) * np_2d_arr.strides[0]).astype(np.intp)
