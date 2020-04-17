#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H
#include <stdint.h>
#include <string.h>

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
}estimate_t;

typedef struct _config_t{
    uint8_t Nx;
    uint8_t Ny;
    uint8_t N_params;
}config_t;

void c_spawn_grid(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, dot2d* out){
    out->x = (((((t->xb-t->xa)*x)/conf->Nx)+t->xa)*y)/conf->Ny;
    out->x += (((((t->xd-t->xc)*x)/conf->Nx)+t->xc)*y)/conf->Ny;
    out->y = (((((t->yd-t->ya)*y)/conf->Ny)+t->ya)*x)/conf->Nx;
    out->y += (((((t->yc-t->yb)*y)/conf->Ny)+t->yb)*x)/conf->Nx;
}

inline void gradient(uint8_t* x, uint8_t* y, int8_t** dimg, config_t* config,int32_t* dout){
    int8_t _dimg = dimg[*x][*y];

}

void c_likelihood(estimate_t* estimate, uint8_t** img,int8_t** dimg, config_t* config, uint32_t* out, int32_t* dout){
    uint8_t x,y;
    dot2d dot;
    *out = 0;
    memset((void*)dout,0,config->N_params*sizeof(dout[0]));

    for(x=0;x<config->Nx;x++){
        for(y=0;y<config->Ny;y++){
            c_spawn_grid(estimate,x,y,config,&dot);
            *out += img[dot.x][dot.y];
        }
    }
}

#endif