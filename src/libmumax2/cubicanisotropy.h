/**
  * @file
  * This file implements the cubic anisotropy field 
  *
  * @author Ben Van de Wiele, Arne Vansteenkiste, Xuanyao (Kelvin) Fong
  */

#ifndef _CUBICANISOTROPY_
#define _CUBICANISOTROPY_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void cubic4AnisotropyAsync(float **hx, float **hy, float **hz, 
                          float **mx, float **my, float **mz,
                          float **K1_map, float **MSat_map, float K1_Mu0Msat_mul, 
                          float **anisC1_mapx, float anisC1_mulx,
                          float **anisC1_mapy, float anisC1_muly,
                          float **anisC1_mapz, float anisC1_mulz,
                          float **anisC2_mapx, float anisC2_mulx,
                          float **anisC2_mapy, float anisC2_muly,
                          float **anisC2_mapz, float anisC2_mulz,
                          CUstream* stream, int Npart);

DLLEXPORT void cubic6AnisotropyAsync(float **hx, float **hy, float **hz, 
                          float **mx, float **my, float **mz,
                          float **K1_map, float **K2_map, float **MSat_map, float K1_Mu0Msat_mul, float K2_Mu0Msat_mul, 
                          float **anisC1_mapx, float anisC1_mulx,
                          float **anisC1_mapy, float anisC1_muly,
                          float **anisC1_mapz, float anisC1_mulz,
                          float **anisC2_mapx, float anisC2_mulx,
                          float **anisC2_mapy, float anisC2_muly,
                          float **anisC2_mapz, float anisC2_mulz,
                          CUstream* stream, int Npart);

DLLEXPORT void cubic8AnisotropyAsync(float **hx, float **hy, float **hz, 
                          float **mx, float **my, float **mz,
                          float **K1_map, float **K2_map, float **K3_map, float **MSat_map, float K1_Mu0Msat_mul, float K2_Mu0Msat_mul, float K3_Mu0Msat_mul, 
                          float **anisC1_mapx, float anisC1_mulx,
                          float **anisC1_mapy, float anisC1_muly,
                          float **anisC1_mapz, float anisC1_mulz,
                          float **anisC2_mapx, float anisC2_mulx,
                          float **anisC2_mapy, float anisC2_muly,
                          float **anisC2_mapz, float anisC2_mulz,
                          CUstream* stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
