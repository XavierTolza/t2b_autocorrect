from itertools import product

from libc.stdint cimport uint32_t, uint8_t,uint16_t
from libcpp cimport bool
from cython.parallel cimport prange
cimport numpy as np
import numpy as pnp
from libc.stdlib cimport  malloc, free
from libc.math cimport cos,sin,sqrt

from t2b.constants import Nb_dots

cdef extern from "likelihood.h":
    ctypedef struct estimate_t:
        uint16_t xa;
        uint16_t xb;
        uint16_t xc;
        uint16_t xd;
        uint16_t ya;
        uint16_t yb;
        uint16_t yc;
        uint16_t yd;
    ctypedef struct config_t:
        uint8_t Nx;
        uint8_t Ny;
        uint8_t N_params;
        uint16_t img_size[2];
    ctypedef struct dot2d:
        uint32_t x;
        uint32_t y;

    void c_spawn_grid(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, dot2d* out)nogil
    void c_spawn_grid_float(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, float* out)nogil

cdef uint32_t Nb_dots_x = Nb_dots[0]
cdef uint32_t Nb_dots_y = Nb_dots[1]
cdef uint32_t Nb_dots_t = Nb_dots[0]*Nb_dots[1]

cpdef config_t get_default_config():
    cdef config_t res
    res.Nx = Nb_dots_x
    res.Ny = Nb_dots_y
    res.N_params = 8
    res.img_size[0] = 0
    res.img_size[1] = 0
    return res


cpdef spawn_grid(uint16_t[::1] estimate):
    res = pnp.zeros((Nb_dots_x*Nb_dots_y,2),dtype=pnp.uint32)
    cdef uint32_t[:, ::1] _res =  res
    cdef uint8_t x,y
    cdef uint32_t i = 0
    cdef config_t conf = get_default_config()

    for x in range(Nb_dots_x):
        for y in range(Nb_dots_y):
            c_spawn_grid(<estimate_t*>(&estimate[0]), x,y,&conf,<dot2d*>(&_res[i,0]))
            i+=1
    return res

cpdef spawn_grid_float(uint16_t[::1] estimate):
    res = pnp.zeros((Nb_dots_x*Nb_dots_y,2),dtype=pnp.float32)
    cdef float[:, ::1] _res =  res
    cdef uint8_t x,y
    cdef uint32_t i = 0
    cdef config_t conf = get_default_config()

    for x in range(Nb_dots_x):
        for y in range(Nb_dots_y):
            c_spawn_grid_float(<estimate_t*>(&estimate[0]), x,y,&conf,<float*>(&_res[i,0]))
            i+=1
    return res