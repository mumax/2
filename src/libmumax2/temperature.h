/**
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _TEMPERATURE_H_
#define _TEMPERATURE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

/// Scales standard normal gaussian noise so that it
/// becomes thermal noise according to Brown.
/// Overwrites input by output.
/// @param alphaMask mask for damping
/// @param temp mask for the temperature
/// @param alphaKB2tempMul multiplier for alpha * temperature * 2 * Boltzmann constant.
/// @param m0VgammaDtMSatMul Mu_zero * cell volume * gyromagnetic ratio * time step * MSat multiplier.
DLLEXPORT void temperature_scaleNoise(float** noise,
			   	float** alpha,
			   	float** temp, float alphaKB2tempMul,
			   	float** mSat, float mu0VgammaDtMSatMul,
			   	CUstream* stream, int Npart);


DLLEXPORT void temperature_scaleAnizNoise(float** hx, float** hy, float** hz,
			   	float** mu_xx, 
			   	float** mu_yy, 
			   	float** mu_zz, 
			   	float** mu_yz, 
			   	float** mu_xz, 
			   	float** mu_xy, 
			   	float** tempMask, 
			   	float** msatMask,
			   	float** msat0T0Mask,
			   	
			   	float muMul_xx,
				float muMul_yy,
				float muMul_zz,
				float muMul_yz,
				float muMul_xz,
				float muMul_xy,
				
				float KB2tempMul, 
			   	float mu0VgammaDtMSatMul,
			   	CUstream* stream, 
			   	int Npart);
#ifdef __cplusplus
}
#endif
#endif
