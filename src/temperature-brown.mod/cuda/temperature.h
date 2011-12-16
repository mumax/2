/**
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _TEMPERATURE_H_
#define _TEMPERATURE_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void temperature_scaleNoise(float** noise,
			   	float** alpha, float alphaMul,
			   	float** temp, float kB2tempMul,
			   	float** mSat, float msatMul,
			   	float mu0VgammaDt,
			   	CUstream* stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
