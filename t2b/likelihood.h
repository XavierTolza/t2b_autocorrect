#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define IMG_INDEX(x,y,c)    ((x) * c->img_size[1] + (y))
#define IMG_INDEX_CORRECT(i,c)    (i< (uint32_t)(c->img_size[0]* c->img_size[1]))
#define DIMG_INDEX(x,y,z,c) (((x) * c->img_size[1]*2) + ((y)*2)+(z))

typedef float est_t;
typedef est_t gradient_t;

typedef struct {
    uint32_t x;
    uint32_t y;
    }dot2d;

typedef struct {
    est_t xa;
    est_t xb;
    est_t xc;
    est_t xd;
    est_t ya;
    est_t yb;
    est_t yc;
    est_t yd;
}estimate_t;

typedef struct {
    uint8_t Nx;
    uint8_t Ny;
    uint8_t N_params;
    uint16_t img_size[2];
}config_t;

void c_spawn_grid_float(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, float* out){
    int16_t Sx1 = t->xb-t->xa;
    int16_t Sx2 = t->xc-t->xd;
    int16_t Sy1 = t->yd-t->ya;
    int16_t Sy2 = t->yc-t->yb;

    float rx = ((float)(x))/(conf->Nx-1);
    float rxi = 1-rx;
    float ry = ((float)(y))/(conf->Ny-1);
    float ryi = 1-ry;

    out[0] = (Sx1*rx + t->xa)*ryi + (Sx2*rx + t->xd)*ry;
    out[1] = (Sy1*ry + t->ya)*rxi + (Sy2*ry + t->yb)*rx;
}

void c_spawn_grid(estimate_t* t,uint8_t x, uint8_t y, config_t* conf, dot2d* out){
    float tmp[2];

    c_spawn_grid_float(t,x,y,conf,tmp);
    out->x = round(tmp[0]);
    out->y = round(tmp[1]);

}

inline void c_gradient(uint8_t* x, uint8_t* y,dot2d* dot, int8_t* dimg, config_t* config,gradient_t* dout){
    float rx = ((float)*x)/((float)config->Nx-1);
    float rxi = 1-rx;
    float ry = ((float)*y)/((float)config->Ny-1);
    float ryi = 1-ry;

    int8_t dimg_x = dimg[DIMG_INDEX(dot->x,dot->y,0,config)];
    int8_t dimg_y = dimg[DIMG_INDEX(dot->x,dot->y,1,config)];

    uint8_t i;
    for (i=0;i<config->N_params/2;i++){
        dout[i] = dimg_x;
    }
    for (i=config->N_params/2;i<config->N_params;i++){
        dout[i] = dimg_y;
    }

    dout[0] *= rxi*ryi; // xa
    dout[1] *= rx*ryi;  // xb
    dout[2] *= rx*ry;   // xc
    dout[3] *= rxi*ry;  // xd

    dout[4] *= rxi*ryi; // ya
    dout[5] *= rx*ryi;  // yb
    dout[6] *= rx*ry;   // yc
    dout[7] *= rxi*ry;  // yd
}

void c_diff_image(uint8_t* img,int8_t* dimg, config_t* conf){
    uint32_t x,y;
    uint32_t xmax = conf->img_size[0]-1;
    uint32_t ymax = conf->img_size[1]-1;
    for (x=1;x<xmax;x++){
        for (y=1;y<ymax;y++){
            dimg[DIMG_INDEX(x,y,0,conf)] = (img[IMG_INDEX(x+1,y,conf)]-img[IMG_INDEX(x-1,y,conf)])/2;
            dimg[DIMG_INDEX(x,y,1,conf)] = (img[IMG_INDEX(x,y+1,conf)]-img[IMG_INDEX(x,y-1,conf)])/2;
        }
    }
}

void c_likelihood(estimate_t* estimate, uint8_t* img,int8_t* dimg, config_t* config, uint32_t* out, float* dout){
    uint8_t x,y;
    dot2d dot;
    *out = 0;
    uint8_t i;
    uint32_t img_index;
    memset((void*)dout,0,config->N_params*sizeof(dout[0]));

    float* tmp = (float*) malloc(config->N_params*sizeof(dout[0]));

    for(x=0;x<config->Nx;x++){
        for(y=0;y<config->Ny;y++){
            c_spawn_grid(estimate,x,y,config,&dot);
            img_index = IMG_INDEX(dot.x,dot.y,config);
            if IMG_INDEX_CORRECT(img_index,config){
                *out += img[img_index];
                c_gradient(&x,&y,&dot,dimg,config,tmp);
                for(i=0;i<config->N_params;i++){
                    dout[i] += tmp[i];
                }
            }
        }
    }
    free(tmp);
}

void c_iterate_estimate(estimate_t* estimate, uint8_t* img,int8_t* dimg, config_t* conf, uint32_t* n_steps,
                        float learning_rate,float exp_average_ratio){
    uint8_t j;
    uint32_t i;
    uint32_t likelihood;
    est_t * _estimate = (est_t*)estimate;

    // Define gradient memory
    uint8_t buffer_size_bytes = (conf->N_params)*sizeof(gradient_t)*2;
    gradient_t* buffer = (gradient_t*) malloc(buffer_size_bytes);
    memset(buffer,0,buffer_size_bytes);

    gradient_t* grad = buffer;
    gradient_t* m = buffer+conf->N_params;



    for(i=0; i<*n_steps; i++){
        // Compute everything
        c_likelihood(estimate,img,dimg,conf,&likelihood,grad);

        for(j=0;j<conf->N_params;j++){
            // Update exp average of gradient
            m[j] *= exp_average_ratio;
            m[j] += (1-exp_average_ratio)*grad[j];

            _estimate[j] += m[j]*learning_rate;
        }
    }

    free(buffer);
}

void c_iterate_many_estimates(estimate_t * estimate, uint32_t n_estimates, uint8_t* img, int8_t* dimg, config_t* conf, uint32_t* n_steps,
                        float learning_rate,float exp_average_ratio){
    uint32_t i;
    for (i=0;i<n_estimates;i++){
        c_iterate_estimate(estimate+i,img,dimg,conf,n_steps,learning_rate,exp_average_ratio);
    }
}
#endif