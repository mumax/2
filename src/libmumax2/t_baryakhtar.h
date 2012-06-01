/**
  * @file
  * This file implements perpendicular Baryakhtar's relaxation
  * See: unpublished W Wang, ..., MD, VVK, MF, HFG (2012)
  * Some CUDA routines are taken from cutil_math.h, subject to NVIDIA licence.
  *
  * @author Mykola Dvornik, Arne Vansteenkiste
  */

#ifndef _T_BARYAKHTAR_H_
#define _T_BARYAKHTAR_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
			 float**  Mx, float**  My, float**  Mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat0,
			 const float msat0Mul,
			 const float lambda,
			 const float lambda_e,
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 const int pbc_x, const int pbc_y, const int pbc_z, 
			 CUstream* stream);
			 
#ifdef __cplusplus
}
#endif
#endif

