/**
  * @file
  * This file implements 6-neighbor exchange using shared memory
  *
  * @author Ben Van de Wiele
  */

#ifndef _EXCHANGE6_2_H_
#define _EXCHANGE6_2_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


DLLEXPORT void exchange6_2Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float** msat, float** aex, float Aex2_mu0MsatMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams);

// Python-style modulo (returns positive int)
// int mod(int a, int b);

#ifdef __cplusplus
}
#endif
#endif
