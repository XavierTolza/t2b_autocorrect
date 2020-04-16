from itertools import product

from libc.stdint cimport uint32_t, uint8_t,uint16_t
from libcpp cimport bool
from cython.parallel cimport prange
cimport numpy as np
import numpy as pnp
from libc.stdlib cimport  malloc, free
from libc.math cimport cos,sin,sqrt

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

cdef struct dot2d:
    uint32_t x;
    uint32_t y;

cpdef dot2d gen_index(uint32_t xi, uint32_t yi,
                      uint32_t ox,uint32_t oy,
                      np.float64_t sx,np.float64_t sy,
                      np.float64_t angle) nogil:
    cdef dot2d res
    cdef np.float64_t c = cos(angle)
    cdef np.float64_t s = sin(angle)
    cdef np.float64_t x = xi * sx
    cdef np.float64_t y = yi * sy
    res.x = <uint32_t> (c*x-s*y+ox)
    res.y = <uint32_t> (s*x+c*y+oy)
    return res

cpdef gen_all_indexes(uint32_t ox,uint32_t oy,np.float64_t sx,np.float64_t sy,np.float64_t angle):
    out = pnp.zeros((Nb_dots_t,2),dtype=pnp.uint32)
    cdef dot2d dot

    for i,(x,y) in enumerate(product(pnp.arange(Nb_dots_x,dtype=pnp.uint32),pnp.arange(Nb_dots_y,dtype=pnp.uint32))):
        dot = gen_index(x,y,ox,oy,sx,sy,angle)
        out[i,:] = [dot.x,dot.y]
    return out

# cpdef np.float64_t _likelihood(uint32_t ox,uint32_t oy,np.float64_t sx,np.float64_t sy,np.float64_t angle,
#                             np.float64_t[:,:] img) nogil:
#     cdef np.float_t res = 0
#     cdef uint32_t xi,yi,N
#     cdef dot2d ixy
#     for xi in range(Nb_dots_x):
#         for yi in range(Nb_dots_y):
#             ixy = gen_index(xi,yi,ox,oy,sx,sy,angle)
#             if ixy.x < img.shape[0] and ixy.y< img.shape[1]:
#                 res += img[ixy.x,ixy.y]
#     return res/Nb_dots_t

# cpdef void likelihood(uint32_t [:] startx,uint32_t [:] starty,
#                       np.float64_t[:] sizex,np.float64_t[:] sizey,
#                       np.float64_t [:] angle,
#                       np.float64_t[:,:] img,
#                       np.float64_t[:] res) nogil:
#     cdef uint32_t shape[5]
#     shape[0] = len(startx)
#     shape[1] = len(starty)
#     shape[2] = len(sizex)
#     shape[3] = len(sizey)
#     shape[4] = len(angle)
#     cdef uint32_t total = shape[0]*shape[1]*shape[2]*shape[3]*shape[4]
#     cdef uint32_t i
#     cdef uint32_t * index
#
#     for i in prange(total,nogil=True):
#         index = <uint32_t *> malloc(sizeof(uint32_t) * 5)
#         c_unravel_index(i,shape,index)
#         res[i] = _likelihood(startx[index[0]],starty[index[1]],
#                              sizex[index[2]],sizey[index[3]],
#                              angle[index[4]],img)
#         free(index)

cdef extern from "likelihood.h":
    uint32_t _likelihood(uint16_t estimate, uint8_t** img) nogil