/**
  * @file
  * This file implements the Zhang-Li spin torque
  * See S. Zhang PRL 93 (2004) pp.127204
  *
  * @author Arne Vansteenkiste
  * @author RŽmy Lassalle-Balier
  */

#ifndef _ZHANG_LI_TORQUE_H_
#define _ZHANG_LI_TORQUE_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// Overwrite h with deltaM(m, h)
///
/// @param mx multi-GPU array of float with the x component of the m vector quantity, non-NULL
/// @param my multi-GPU array of float with the y component of the m vector quantity, non-NULL
/// @param mz multi-GPU array of float with the z component of the m vector quantity, non-NULL
/// @param hx multi-GPU array of float with the x component of the effective field vector quantity, non-NULL
/// @param hy multi-GPU array of float with the y component of the effective field vector quantity, non-NULL
/// @param hz multi-GPU array of float with the z component of the effective field vector quantity, non-NULL
/// @param alpha multi-GPU array of float with the damping quantity, non-NULL
/// @param beta {=b(1+alpha*xi)} multi-GPU array of float with the beta quantity, non-NULL
/// @param epsilon {=b(xi-alpha)} multi-GPU array of float with the epsilon quantity, non-NULL
/// @param ux {=U_spintorque/(2*cellsize[X])} float with the x component of the u vector quantity, non-NULL
/// @param uy {=U_spintorque/(2*cellsize[X])} float with the y component of the u vector quantity, non-NULL
/// @param uz {=U_spintorque/(2*cellsize[X])} float with the z component of the u vector quantity, non-NULL
/// @param jx multi-GPU array of float with the x component of the j vector quantity, non-NULL
/// @param jy multi-GPU array of float with the y component of the j vector quantity, non-NULL
/// @param jz multi-GPU array of float with the z component of the j vector quantity, non-NULL
/// @param dt_gilb dt * gilbert factor
/// @param N0 number of elements along x axis
/// @param N1Part number of elements along y axis dealt with on this device (the structure is sliced for multiGPU purpose along Y-axis)
__declspec(dllexport) void spintorque_deltaMAsync(float** mx, float** my, float** mz,
        				   float** hx, float** hy, float** hz,
                           float** alpha,
                           float** bj,
                           float** cj,
                           float** Msat,
                           float ux, float uy, float uz,
                           float** jx, float** jy, float** jz,
                           float dt_gilb,
						   CUstream* stream,
                           int N0, int N1Part, int N2);

#ifdef __cplusplus
}
#endif
#endif
