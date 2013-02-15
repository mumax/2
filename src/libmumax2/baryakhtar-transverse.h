/**
  * @file
  * This file implements perpendicular Baryakhtar's relaxation
  * See: unpublished W Wang, ..., MD, VVK, MF, HFG (2012)
  * 
  * @author Mykola Dvornik
  */
  
#ifndef _BARYAKHTAR_TRANSVERSE_H_
#define _BARYAKHTAR_TRANSVERSE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void baryakhtar_transverse_async(float** tx, float**  ty, float**  tz, 
			 float**  Mx, float**  My, float**  Mz, 
			 float**  hx, float**  hy, float**  hz,
			 
			 float** msat0T0,
			 
			 float** mu_xx,
			 float** mu_yy,
			 float** mu_zz,
			 float** mu_yz,
			 float** mu_xz,
			 float** mu_xy,
			 
			 const float muMul_xx,
			 const float muMul_yy,
			 const float muMul_zz,
			 const float muMul_yz,
			 const float muMul_xz,
			 const float muMul_xy,
			 
			 CUstream* stream,
			 int Npart);
			 
#ifdef __cplusplus
}
#endif
#endif

