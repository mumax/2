#ifndef _ADD_H_
#define _ADD_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void add(float** dst, float** a, float** b, CUstream* stream, int Ndev, int Npart);


#ifdef __cplusplus
}
#endif
#endif
