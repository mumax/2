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
             float** l,
			 float**  mx, float**  my, float**  mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat,
			 float**  AexMsk,
			 float**  alphaMsk,
			 const float alphaMul,
			 const float pred,
			 const float pre,
			 const float pret,
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 const int pbc_x, const int pbc_y, const int pbc_z, 
			 CUstream* stream);
			 
#ifdef __cplusplus
}
#endif
#endif

