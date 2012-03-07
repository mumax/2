/**
  * @file
  * This file implements Slonczewski spin torque
  * See Slonczewski JMMM 159 (1996) L1-L7
  *
  * @author Graham Rowlands, Arne Vansteenkiste
  */

#ifndef _SLONCZEWSKI_TORQUE_H_
#define _SLONCZEWSKI_TORQUE_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void slonczewski_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** px, float** py, float** pz,
			 float pxMul, float pyMul, float pzMul,
			 float aj, float bj, float Pol,
			 float** jx, float jxMul,
			 int NPart, 
			 CUstream* stream);

#ifdef __cplusplus
}
#endif
#endif
