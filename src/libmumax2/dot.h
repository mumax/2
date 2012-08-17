/**
  * @file
  * This file implements the torque according to Landau-Lifshitz.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _DOT_H_
#define _DOT_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// calculates the dot product
DLLEXPORT void dotAsync(float** dst, float** ax, float** ay, float** az, float** bx, float** by, float** bz, CUstream* stream, int Npart);


#ifdef __cplusplus
}
#endif
#endif
