/**
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _COPYPAD_H_
#define _COPYPAD_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Copy+zero pad in Z-direction (user X)
/// @param dst: destination arrays
/// @param D2: dst Z size, >= S2
/// @param src: source arrays
/// @param S0: source X size, same as dst X size
/// @param S1: source Y size per GPU, same as dst Y size
/// @param S2: source Z size , <= D2
DLLEXPORT void copyPadZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, CUstream* streams);

/// Copy+zero pad a 3D matrix in all directions: only to be used when 1 device is used.
/// @param dst: destination arrays
/// @param D0: dst X size, >= S0
/// @param D1: dst Y size, >= S1
/// @param D2: dst Z size, >= S2
/// @param src: source arrays
/// @param S0: source X size , <= D0
/// @param S1: source Y size , <= D1
/// @param S2: source Z size , <= D2
/// @param Ncomp: number of array components
DLLEXPORT void copyPad3DAsync(float** dst, int D0, int D1, int D2, float** src, int S0, int S1, int S2, int Ncomp, CUstream* streams);


/// Insert from src into a block in dst
/// E.g.:
///	2x2 src, block = 1, 2x6 dst:
///	[ 0 0  S1 S2  0 0 ]
///	[ 0 0  S3 S4  0 0 ]
DLLEXPORT void insertBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams);

/// Put an array to zero with (sub)sizes [NO, N1part, N2]
DLLEXPORT void zeroArrayAsync(float **A, int N, CUstream *streams);

// /// Put a part of an array on 1 GPU to zero
// DLLEXPORT void zeroArrayPartAsync(float **A, int length, int dev, CUstream streams){


/// Extract from src a block to dst
/// E.g.:
/// 2x2 dst, block = 1, 2x6 src:
/// [ 0 0  D1 D2  0 0 ]
/// [ 0 0  D3 D4  0 0 ]
DLLEXPORT void extractBlockZAsync(float **dst, int D0, int D1Part, int D2, float **src, int S2, int block, CUstream *streams);

#ifdef __cplusplus
}
#endif
#endif
