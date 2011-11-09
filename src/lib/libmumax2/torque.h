/**
  * @file
  * This file implements the torque according to Landau-Lifshitz.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _TORQUE_H_
#define _TORQUE_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// calculates the reduced Landau-Lifshitz torque τ, in units gamma0*Msat:
///	d m / d t = gamma0 * Msat * τ  
/// Note: the unit of gamma0 * Msat is 1/time.
/// Thus:
///	τ = (m x h) - α m  x (m x h)
/// with:
///	h = H / Msat
/// 
/// @param tx, ty, tz multi-GPU array to store torque
/// @param mx, my, mz multi-GPU array with magnetization components, non-NULL
/// @param mx, my, mz multi-GPU array with reduced effective field components (H/Msat), non-NULL
/// @param alpha_map multi-GPU array with space-dependent scaling factors for alpha. NULL is interpreted as all 1's
/// @param alpha damping coefficient
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
void torqueAsync(float** tx, float** ty, float** tz, float** mx, float** my, float** mz, float** hx, float** hy, float** hz, float** alpha_map, float alpha_mul, CUstream* stream, int Npart);


#ifdef __cplusplus
}
#endif
#endif
