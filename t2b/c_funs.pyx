from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint8_t, uint32_t
from libcpp cimport bool
from cython.parallel cimport prange
cimport numpy as np
import numpy as pnp
from libc.stdlib cimport abort, malloc, free
from libc.math cimport cos,sin

from t2b.constants import Nb_dots

cdef uint32_t Nb_dots_x = Nb_dots[0]
cdef uint32_t Nb_dots_y = Nb_dots[1]
cdef uint32_t Nb_dots_t = Nb_dots[0]*Nb_dots[1]

cdef void c_unravel_index(uint32_t i, uint32_t* shape, uint32_t* res) nogil:
    cdef uint32_t counter = i
    cdef uint32_t current_shape
    cdef uint32_t mod
    cdef uint32_t inv_index
    cdef uint32_t j

    for j in range(5):
        inv_index = 4 - j
        current_shape = shape[inv_index]
        mod = counter % current_shape
        res[inv_index] = mod
        counter -= mod
        counter = counter / current_shape

cpdef unravel_index(i,shape):
    s = pnp.array(shape,dtype=pnp.uint32)
    cdef uint32_t[:] _s = s
    res = pnp.zeros(s.size,dtype=pnp.uint32)
    cdef uint32_t[:] _res = res
    c_unravel_index(i,&_s[0],&_res[0])
    return res

cdef np.float64_t _likelihood(uint32_t ox,uint32_t oy,uint32_t sx,uint32_t sy,np.float64_t angle,
                            np.float64_t[:,:] img) nogil:
    cdef np.float_t res = 0
    cdef uint32_t xi,yi,ix,iy,N
    for xi in range(Nb_dots_x):
        for yi in range(Nb_dots_y):
            ix = <uint32_t> (cos(angle)*sx*xi+ox)
            iy = <uint32_t> ((sin(angle)*sy*yi+oy))
            if ix < img.shape[0] and iy< img.shape[1]:
                res += img[ix,iy]
    return res/Nb_dots_t


cpdef void likelihood(uint32_t [:] start, uint32_t[:] size, np.float64_t [:] angle,np.float64_t[:,:] img,
                      np.float64_t[:] res) nogil:
    cdef uint32_t shape[5]
    shape[0] = shape[1] = len(start)
    shape[2] = shape[3] = len(size)
    shape[4] = len(angle)
    cdef uint32_t total = shape[0]*shape[1]*shape[2]*shape[3]*shape[4]
    cdef uint32_t i
    cdef uint32_t * index

    for i in prange(total,nogil=True):
        index = <uint32_t *> malloc(sizeof(uint32_t) * 5)
        c_unravel_index(i,shape,index)
        res[i] = _likelihood(start[index[0]],start[index[1]],size[index[2]],size[index[3]],angle[index[4]],img)
        free(index)
