from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint8_t, uint32_t
from libcpp cimport bool
from cython.parallel cimport prange
import numpy as np

cdef extern from "lib.h":
    # bluetooth_adv_packet c_parse_bytes(vector[uint8_t] data, uint8_t channel, bool swap_bytes, bool _dewhitening)
    float c_likelihood(uint32_t startx, uint32_t starty, uint32_t sizex, uint32_t sizey, float angle,
                       vector[vector[float]] img) nogil

cpdef float likelihood(uint32_t startx, uint32_t starty, uint32_t sizex, uint32_t sizey, float angle,
                       vector[vector[float]] img) nogil:
    return c_likelihood(startx, starty, sizex, sizey, angle, img)

cpdef float vectorized_likelihood(vector[uint32_t] start, vector[uint32_t] size, vector[float] angle,
                                  vector[vector[float]] img):
    res_shape = tuple([i.size for i in [start, start, size, size, angle]])
    res = np.zeros(np.prod(res_shape))
    return res
