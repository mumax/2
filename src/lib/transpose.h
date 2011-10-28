
/**
 * @file
 * Transposition of a tensor of complex numbers
 *
 * @note Be sure not to use nvcc's -G flag, as this
 * slows down these functions by an order of magnitude
 *
 * @author Arne Vansteenkiste
 */
#ifndef _TRANSPOSE_H
#define _TRANSPOSE_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// Per-GPU 2D complex matrix partial transpose. 
/// Input size: N0xN1xN2 reals (N1 x N2/2 complex) numbers per GPU.
///	Output size:N0xN2/2xN1*2 reals (N2/2 x N1 complex) numbers per GPU.
/// @note The result is not yet the transposed full matrix, data still has to be exchanged between GPUs.
void transposeComplexYZAsyncPart(float** output, float** input, int N0, int N1, int N2, CUstream* stream);

/// Swaps the Y and Z dimension of an array of complex numbers in interleaved format
//void transposeComplexXZ(float *input, float *output, int N0, int N1, int N2);

/// 2D real matrix transpose. Input size: N1 x N2, Output size: N2 x N1
// void gpu_transpose(float *input, float *output, int N1, int N2);

/// Swaps the Y and Z dimension of an array of complex numbers in interleaved format
// void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2);

/// Swaps the X and Z dimension of an array of complex numbers in interleaved format
// void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2);



#ifdef __cplusplus
}
#endif
#endif
