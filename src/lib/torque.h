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


void torqueAsync(float** tx, float** ty, float** tz, float** mx, float** my, float** mz, float** hx, float** hy, float** hz, float** alpha_map, float alpha_mul, CUstream* stream, int Npart);


#ifdef __cplusplus
}
#endif
#endif
