from itertools import product

from libc.stdint cimport uint32_t, uint8_t,uint16_t,int8_t
from libcpp cimport bool
from cython.parallel cimport prange
cimport numpy as np
import numpy as pnp
from libc.stdlib cimport  malloc, free
from libc.math cimport cos,sin,sqrt

from t2b.constants import Nb_dots

estimate_dtype = pnp.float32

cdef extern from "likelihood.h":
    ctypedef float est_t
    ctypedef est_t gradient_t
    ctypedef struct estimate_t:
        est_t xa;
        est_t xb;
        est_t xc;
        est_t xd;
        est_t ya;
        est_t yb;
        est_t yc;
        est_t yd;
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
    void c_likelihood(estimate_t* estimate, uint8_t* img, int8_t* dimg, config_t* config, uint32_t* out, float* dout)nogil
    void c_diff_image(uint8_t* img,int8_t* dimg, config_t* conf)nogil except+
    void c_gradient(uint8_t* x, uint8_t* y,dot2d* dot, int8_t* dimg, config_t* config,float* dout)nogil
    void c_iterate_estimate(estimate_t* estimate, uint8_t* img,int8_t* dimg, config_t* conf, uint8_t* n_steps,
                        float learning_rate)nogil

cdef uint32_t Nb_dots_x = Nb_dots[0]
cdef uint32_t Nb_dots_y = Nb_dots[1]
cdef uint32_t Nb_dots_t = Nb_dots[0]*Nb_dots[1]

cpdef config_t get_default_config(image=None)except+:
    cdef config_t res
    res.Nx = Nb_dots_x
    res.Ny = Nb_dots_y
    res.N_params = 8
    if image is not None:
        res.img_size[0] = image.shape[0]
        res.img_size[1] = image.shape[1]
    else:
        res.img_size[0] = 0
        res.img_size[1] = 0
    return res


cpdef spawn_grid(est_t[::1] estimate):
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

cpdef spawn_grid_float(est_t[::1] estimate):
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

cpdef likelihood(est_t[::1] estimate, img):
    assert img.ndim==2
    cdef uint8_t[::1] _img = pnp.ascontiguousarray(img.ravel())
    dimg_python = pnp.zeros(pnp.prod(img.shape+(2,)),dtype=pnp.int8)
    cdef int8_t[::1] dimg = dimg_python
    cdef int8_t* dimg_p = &(dimg[0])
    cdef uint8_t* img_p = &(_img[0])
    cdef config_t config = get_default_config(img)
    c_diff_image(img_p,dimg_p,&config)
    cdef uint32_t res;
    dres = pnp.zeros(config.N_params,dtype=pnp.float32)
    cdef float[:] _dres = dres

    c_likelihood(<estimate_t*>(&estimate[0]),img_p,dimg_p,&config,&res,&_dres[0])

    return res, dres

cpdef void gradient(uint8_t x, uint8_t y, dot2d dot, dimg,float[::1] out)except+:
    cdef int8_t[::1] _dimg = dimg.ravel()
    cdef config_t conf = get_default_config(dimg)
    c_gradient(&x,&y,&dot,&_dimg[0],&conf,&out[0])

cpdef diff_image(uint8_t[:,::1] img) except+:
    cdef config_t conf = get_default_config(img)
    res = pnp.zeros(img.size*2,dtype=pnp.int8)
    cdef int8_t[::1] _res = res
    # img_r = img.ravel()
    # cdef uint8_t[::1] _img = img_r
    cdef uint8_t* img_p = &img[0,0]

    c_diff_image(img_p,&_res[0],&conf)

    return res.reshape((img.shape[0],img.shape[1],2))

cpdef iterate_estimate(est_t[::1] estimate, img,int8_t[::1] dimg, uint8_t n_steps, float learning_rate):
    cdef config_t conf = get_default_config(img)
    cdef uint8_t[:] _img = img.ravel()
    c_iterate_estimate(<estimate_t*>&estimate[0],&_img[0],&dimg[0],&conf,&n_steps,learning_rate)

    # res = pnp.zeros_like(estimate)
    # cdef uint8_t i
    # for i in range(conf.N_params):
    #     res[i] = estimate[i]
    #
    # return res
