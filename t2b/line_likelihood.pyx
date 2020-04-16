from itertools import product

from libc.stdint cimport uint32_t
from libcpp cimport bool
from cython.parallel cimport prange
cimport numpy as np
import numpy as pnp
from libc.stdlib cimport  malloc, free
from libc.math cimport cos,sin,sqrt

from t2b.constants import Nb_dots

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

cpdef line_likelihood(angle,radius,img):
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