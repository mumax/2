/**
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _COPYPAD_H_
#define _COPYPAD_H_

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


/// Copy from src into a block in dst
/// E.g.:
///	2x2 source, block = 1, 2x6 dst:
///	[ 0 0  S1 S2  0 0 ]
///	[ 0 0  S3 S4  0 0 ]
void copyBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams);


#ifdef __cplusplus
}
#endif
#endif
