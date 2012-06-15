/**
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _LIMITER_H_
#define _LIMITER_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// Normalizes a vector array if length goes beyond the limit.
/// @param mx, my, mz: Components of vector array to normalize
/// @param limit: The uper limit of the vector length
DLLEXPORT void limiterAsync(float** Mx, float** My, float** Mz,
                               float limit,
                               CUstream* stream, int Npart);

#ifdef __cplusplus
}
#endif

#endif
