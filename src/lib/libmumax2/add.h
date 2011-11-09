/**
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _ADD_H_
#define _ADD_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = a[i] + b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
void addAsync(float** dst, float** a, float** b, CUstream* stream, int Npart);

/// Multiply-add: dst[i] = a[i] + mulB * b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
void maddAsync(float** dst, float** a, float** b, float mulB, CUstream* stream, int Npart);




#ifdef __cplusplus
}
#endif
#endif
