/**
  * @file
  * This file implements Slonczewski spin torque
  * See Slonczewski JMMM 159 (1996) L1-L7
  *
  * @author Graham Rowlands
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
			 float** alpha, float** Msat,
			 float gamma, float aj, float bj, float Pol,
			 float **curr, 
			 int N0, int N1Part, int N2, 
			 CUstream* stream);
#ifdef __cplusplus
}
#endif
#endif
