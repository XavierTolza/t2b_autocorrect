#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H


typedef struct _dot2d{
    uint32_t x;
    uint32_t y;
    }dot2d;

typedef struct _estimate_t{
    uint16_t xa;
    uint16_t xb;
    uint16_t xc;
    uint16_t xd;
    uint16_t ya;
    uint16_t yb;
    uint16_t yc;
    uint16_t yd;
}estimate_t

typedef struct _config_t{
    uint8_t Nx;
    uint8_t Ny;
}config_t

void c_spawn_grid(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, dot2d* out){
    out.x = (((((t->xb-t->xa)*x)/conf->Nx)+t->xa)*y)/conf->Ny;
    out.x += (((((t->xd-t->xc)*x)/conf->Nx)+t->xc)*y)/conf->Ny;
    out.y = (((((t->yd-t->ya)*y)/conf->Ny)+t->ya)*x)/conf->Nx;
    out.y += (((((t->yc-t->yb)*y)/conf->Ny)+t->yb)*x)/conf->Nx;
}

uint32_t c_likelihood(uint16_t* estimate, uint8_t** img){
    uint8_t x,y;
    return 0;
}

#endif