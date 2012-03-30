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

  __declspec(dllexport) void slonczewski_async(float** sttx, float** stty, float** sttz,  ///< output
			 float** mx, float** my, float** mz,  ///< magnetization
			 float** px, float** py, float** pz, ///< fixed layer
			 float pxMul, float pyMul, float pzMul, ///< multipliers for fixed layer
			 float aj, float bj, float Pol, 
			 float** jx, float jxMul, ///< out-of-plane current density
			 int NPart, 
			 CUstream* stream);

#ifdef __cplusplus
}
#endif
#endif
