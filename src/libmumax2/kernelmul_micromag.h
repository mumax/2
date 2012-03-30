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


/// @param fftMx fftMy fftMz Fourier-transformed magnetization
/// @param fftKxx... Fourier-transformed convolution kernel, symmetric and purely real. Size is half the size of fftM*!
/// @param partLen3D number of floats (not complex) PER GPU, PER fftM* COMPONENT
/// |Hx|   |Kxx Kxy Kxz|   |Mx|
/// |Hy| = |Kxy Kyy Kyz| * |My|
/// |Hz|   |Kxz Kyz Kzz|   |Mz|
__declspec(dllexport) void kernelMulMicromag3DAsync(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz,
                              float** fftKyz, float** fftKxz, float** fftKxy,
                              CUstream* stream, int partLen3D);

#ifdef __cplusplus
}
#endif
#endif
