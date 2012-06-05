/**
  * @file
  *
  * @author Ben Van de Wiele
  */

#ifndef _KERNELMUL_MICROMAG2_H_
#define _KERNELMUL_MICROMAG2_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// @param fftMx fftMy fftMz Fourier-transformed magnetization
/// @param fftKxx... Fourier-transformed convolution kernel, symmetric and purely real. Symmetry is fully exploited!
/// @param outx outy outz output arrays, can be the input arrays
/// @param partSize number of floats (not complex) PER GPU, PER fftM* COMPONENT
/// |outx|   |Kxx Kxy Kxz|   |Mx|
/// |outy| = |Kxy Kyy Kyz| * |My|
/// |outz|   |Kxz Kyz Kzz|   |Mz|
DLLEXPORT void kernelMulMicromag3D2Async(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz,
                              float** fftKyz, float** fftKxz, float** fftKxy,
                              float** outx, float** outy, float** outz,
                              CUstream* stream, int* partSize);



/// @param fftMx fftMy fftMz Fourier-transformed magnetization
/// @param fftKxx... Fourier-transformed convolution kernel, symmetric and purely real. Symmetry is fully exploited!
/// @param outx outy outz output arrays, can be the input arrays
/// @param partSize number of floats (not complex) PER GPU, PER fftM* COMPONENT
/// |outx|   |Kxx Kxy Kxz|   |Mx|
/// |outy| = |Kxy Kyy Kyz| * |My|
/// |outz|   |Kxz Kyz Kzz|   |Mz|
DLLEXPORT void kernelMulMicromag2D2Async(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz, float** fftKyz,
                              float** outx, float** outy, float** outz,
                              CUstream* stream, int* partSize);

#ifdef __cplusplus
}
#endif
#endif
