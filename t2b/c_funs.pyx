from itertools import product

from libc.stdint cimport uint32_t
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

cpdef np.float64_t _likelihood(uint32_t ox,uint32_t oy,np.float64_t sx,np.float64_t sy,np.float64_t angle,
                            np.float64_t[:,:] img) nogil:
    cdef np.float_t res = 0
    cdef uint32_t xi,yi,N
    cdef dot2d ixy
    for xi in range(Nb_dots_x):
        for yi in range(Nb_dots_y):
            ixy = gen_index(xi,yi,ox,oy,sx,sy,angle)
            if ixy.x < img.shape[0] and ixy.y< img.shape[1]:
                res += img[ixy.x,ixy.y]
    return res/Nb_dots_t


cpdef void likelihood(uint32_t [:] startx,uint32_t [:] starty,
                      np.float64_t[:] sizex,np.float64_t[:] sizey,
                      np.float64_t [:] angle,
                      np.float64_t[:,:] img,
                      np.float64_t[:] res) nogil:
    cdef uint32_t shape[5]
    shape[0] = len(startx)
    shape[1] = len(starty)
    shape[2] = len(sizex)
    shape[3] = len(sizey)
    shape[4] = len(angle)
    cdef uint32_t total = shape[0]*shape[1]*shape[2]*shape[3]*shape[4]
    cdef uint32_t i
    cdef uint32_t * index

    for i in prange(total,nogil=True):
        index = <uint32_t *> malloc(sizeof(uint32_t) * 5)
        c_unravel_index(i,shape,index)
        res[i] = _likelihood(startx[index[0]],starty[index[1]],
                             sizex[index[2]],sizey[index[3]],
                             angle[index[4]],img)
        free(index)

cdef bool c_point_is_in_image(np.float64_t x,np.float64_t y,uint32_t xmax,uint32_t ymax)nogil:
    if x<0 or y<0:
        return False
    if x>xmax or y>ymax:
        return False
    return True

cdef void c_line_find_coordinates(np.float64_t angle,np.float64_t radius, np.float64_t[:,:] img, np.float64_t* out)nogil:
    cdef uint32_t xmax = img.shape[0]
    cdef uint32_t ymax = img.shape[1]
    cdef np.float64_t x1,y1,x2,y2,s,c
    s = sin(angle)
    c=cos(angle)
    cdef np.float64_t* intersection = [-1,-1,-1,-1,-1,-1,-1,-1]
    if c!=0:
        intersection[2] = radius/c
        intersection[3] = 0
        intersection[6] = (radius-ymax*s)/c
        intersection[7] = ymax
    if s!=0:
        intersection[0] = 0
        intersection[1] = radius/s
        intersection[4] = xmax
        intersection[5] = (radius-xmax*c)/s
    cdef uint32_t i,j,k

    for i in range(4):
        x1 = intersection[i*2]
        y1 = intersection[i*2+1]
        if c_point_is_in_image(x1,y1,xmax,ymax):
            for j in range(4):
                if j!=i:
                    x2 = intersection[j*2]
                    y2 = intersection[j*2+1]
                    if c_point_is_in_image(x2,y2,xmax,ymax):
                        out[0] = x1
                        out[1] = y1
                        out[2] = x2
                        out[3] = y2
                        return

cpdef line_find_coordinates(np.float64_t angle,np.float64_t radius, np.float64_t[:,:] img):
    res=pnp.zeros(4)
    cdef np.float64_t[:] tmp = res
    cdef np.float64_t* _res = &tmp[0]
    c_line_find_coordinates(angle,radius,img,_res)
    return res

cdef np.float64_t c_line_likelihood(np.float64_t angle,np.float64_t radius, np.float64_t[:,:] img) nogil:
    # On trouve les points d'intersection
    cdef np.float64_t* coord = [0,0,0,0]
    c_line_find_coordinates(angle,radius,img,coord)
    cdef np.float64_t x1,y1,x2,y2,ratio
    cdef uint32_t distance,x,y
    x1,y1,x2,y2 = coord[0],coord[1],coord[2],coord[3]
    cdef np.float64_t result=0
    cdef uint32_t xmax = img.shape[0]
    cdef uint32_t ymax = img.shape[1]

    distance = <uint32_t>(sqrt(((x2-x1)**2+(y2-y1)**2)/2.0))
    if distance>10:
        # On a les coordonnées de la ligne, on peut sélectionner les points intermédiaires
        for k in range(1,distance):
            ratio = (<np.float64_t>k)/<np.float64_t>(distance)
            x = <uint32_t>(x2*ratio+(1-ratio)*x1)
            y = <uint32_t>(y2*ratio+(1-ratio)*y1)
            # print(([angle,radius],k,[x1,y1],[x2,y2],[x,y],result))
            if x < xmax and y < ymax:
                result += img[x,y]/<np.float64_t>(distance)
        return result

    return 0

cdef void c_multi_line_likelihood(np.float64_t[:] angle,np.float64_t[:] radius, np.float64_t[:,:] img, np.float64_t[:] out)nogil:
    cdef uint32_t total=angle.shape[0]
    cdef uint32_t i

    # for i in range(total):
    for i in prange(total,nogil=True):
        out[i] = c_line_likelihood(angle[i],radius[i],img)

cpdef line_likelihood(angle,radius,img,transform_angle_radius=True):
    cdef uint32_t total = angle.size
    assert total == radius.size
    res = pnp.zeros(total,dtype=pnp.float64)
    cdef np.float64_t[:] _res=res
    cdef np.float64_t[:,:] _img = img.astype(pnp.float64)
    cdef np.float64_t [::1] _a = angle.astype(pnp.float64)
    cdef np.float64_t [::1] _b = radius.astype(pnp.float64)
    cdef uint32_t i

    c_multi_line_likelihood(_a,_b,_img,_res)
    return res
