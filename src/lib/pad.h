/**
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _PAD_H_
#define _PAD_H_

#include <cuda.h>

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
void copyPadZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, CUstream* streams);


void copyBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams);


/// Concatenate two matrices in Z-direction (user X)
/// @code
/// 	src1[0,0]  src1[0,1]  src2[0,0]  src2[0,1]
/// 	src1[1,0]  src1[1,1]  src2[1,0]  src2[1,1]
/// @code
/// @param dst: destination arrays
/// @param D2: dst Z size, >= S2
/// @param src1: source arrays
/// @param src2: source arrays
/// @param S0: source X size, same as dst X size
/// @param S1: source Y size per GPU, same as dst Y size
/// @param S2: source Z size , <= D2
void combineZAsync(float** dst, int D2, float** src1, float** src2, int S0, int S1Part, int S2, CUstream* streams);

#ifdef __cplusplus
}
#endif
#endif
