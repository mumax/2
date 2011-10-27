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

#ifdef __cplusplus
}
#endif
#endif
