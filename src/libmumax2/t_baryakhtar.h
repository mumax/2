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
			 float** lambda_xx,
			 float** lambda_yy,
			 float** lambda_zz,
			 float** lambda_yz,
			 float** lambda_xz,
			 float** lambda_xy,
			 
			 float** lambda_e_xx,
			 float** lambda_e_yy,
			 float** lambda_e_zz,
			 float** lambda_e_yz,
			 float** lambda_e_xz,
			 float** lambda_e_xy,
			 
			 const float msat0Mul,
			 const float lambdaMul_xx,
			 const float lambdaMul_yy,
			 const float lambdaMul_zz,
			 const float lambdaMul_yz,
			 const float lambdaMul_xz,
			 const float lambdaMul_xy,
			 
			 const float lambda_eMul_xx,
			 const float lambda_eMul_yy,
			 const float lambda_eMul_zz,
			 const float lambda_eMul_yz,
			 const float lambda_eMul_xz,
			 const float lambda_eMul_xy,
			 
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 const int pbc_x, const int pbc_y, const int pbc_z, 
			 CUstream* stream);
			 
#ifdef __cplusplus
}
#endif
#endif

