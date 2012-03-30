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

/// Scales standard normal gaussian noise so that it
/// becomes thermal noise according to Brown.
/// Overwrites input by output.
/// @param alphaMask mask for damping
/// @param temp mask for the temperature
/// @param alphaKB2tempMul multiplier for alpha * temperature * 2 * Boltzmann constant.
/// @param m0VgammaDtMSatMul Mu_zero * cell volume * gyromagnetic ratio * time step * MSat multiplier.
__declspec(dllexport) void temperature_scaleNoise(float** noise,
			   	float** alpha,
			   	float** temp, float alphaKB2tempMul,
			   	float** mSat, float mu0VgammaDtMSatMul,
			   	CUstream* stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
