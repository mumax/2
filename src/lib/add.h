#ifndef _LIBMUMAX2_H
#define _LIBMUMAX2_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void add(float** dst, float** a, float** b, CUstream* stream, int Ndev, int Npart);


#ifdef __cplusplus
}
#endif
#endif
