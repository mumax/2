/**
  * @file
  * This file implements Zhang-Li spin torque
  * See S. Zhang and Z. Li, PRL 93, 127204 (2004)
  * Some CUDA routines are taken from cutil_math.h, subject to NVIDIA licence.
  *
  * @author Mykola Dvornik, Arne Vansteenkiste
  */

#ifndef _ZHANGLINEW_TORQUE_H_
#define _ZHANGLINEW_TORQUE_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport)  void zhangli_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** jx, float** jy, float** jz,
			 const float pred, const float pret,
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz, 
			 int NPart,
			 CUstream* stream);
			 
#ifdef __cplusplus
}
#endif
#endif

