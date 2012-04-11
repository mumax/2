
/**
 * @file
 * Transposition of a tensor of complex numbers
 *
 * @note Be sure not to use nvcc's -G flag, as this
 * slows down these functions by an order of magnitude
 *
 * @author Arne Vansteenkiste, Ben Van de Wiele
 */
#ifndef _TRANSPOSE_H
#define _TRANSPOSE_H

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

/// Per-GPU 2D complex matrix partial transpose. 
/// Input size: N0xN1xN2 reals (N1 x N2/2 complex) numbers per GPU.
///	Output size:N0xN2/2xN1*2 reals (N2/2 x N1 complex) numbers per GPU.
/// @note The result is not yet the transposed full matrix, data still has to be exchanged between GPUs.
DLLEXPORT void transposeComplexYZAsyncPart(float** output, float** input, int N0, int N1, int N2, CUstream* stream);
DLLEXPORT void transposeComplexYZSingleGPUFWAsync(float** output, float** input, int N0, int N1, int N2, int N2out, CUstream* stream);
DLLEXPORT void transposeComplexYZSingleGPUINVAsync(float** output, float** input, int N0, int N1, int N2, int N1out, CUstream* stream);


#ifdef __cplusplus
}
#endif
#endif
