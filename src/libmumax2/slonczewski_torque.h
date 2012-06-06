/**
  * @file
  * This file implements Slonczewski spin torque
  * See Slonczewski JMMM 159 (1996) L1-L7 and 
  *
  * @author Mykola Dvorni, Graham Rowlands, Arne Vansteenkiste
  */

#ifndef _SLONCZEWSKI_TORQUE_H_
#define _SLONCZEWSKI_TORQUE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

  DLLEXPORT void slonczewski_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** msat,
			 float** px, float** py, float** pz,
			 float** jx, float** jy, float** jz,
			 float** alphamsk,
             float** t_flmsk,
			 float pxMul, float pyMul, float pzMul,
			 float jxMul, float jyMul, float jzMul,
			 float lambda2, float beta_prime, float pre_field,
			 float meshSizeX,float meshSizeY, float meshSizeZ, 
			 float alphaMul,
             float t_flMul,
			 int NPart, 
			 CUstream* stream);

#ifdef __cplusplus
}
#endif
#endif
