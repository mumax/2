/**
  * @file
  *
  * @author Arne Vansteenkiste, Ben Van de Wiele
  */

#ifndef _KERNELMUL_MICROMAG_H_
#define _KERNELMUL_MICROMAG_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// |Hx|   |Kxx Kxy Kxz|   |Mx|
/// |Hy| = |Kxy Kyy Kyz| * |My|
/// |Hz|   |Kxz Kyz Kzz|   |Mz|
void kernelMulMicromag3DAsync(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz,
                              float** fftKyz, float** fftKxz, float** fftKxy,
                              CUstream* stream, int nRealNumbers);

#ifdef __cplusplus
}
#endif
#endif
