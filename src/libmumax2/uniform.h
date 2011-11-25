/**
  * @file
  * This file implements the initialisation of a given quantity based on region system.
  * Value must be uniform in each region.
  *
  * @author RŽmy Lassalle-Balier
  */

#ifndef _QUANT_INIT_H_
#define _QUANT_INIT_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// Initialise the scalar quantity S using region system (value is uniform in each region)
///
/// @param S multi-GPU array of float with the scalar quantity to initialise, non-NULL
/// @param regions multi-GPU array of float with region definition, non-NULL
/// @param initValues multi-GPU array of float with initial value of the S quantity per region, non-NULL. This array must be ordered by region index value.
/// @note The scalar value must be uniform in eqch region.
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
void initScalarQuantUniformRegionAsync(float** S, float** regions, float* initValues, int initValNum, CUstream* stream, int Npart);

/// Initialise the vector quantity S = (Sx,Sy,Sz) using region system (value is uniform in each region)
///
/// @param Sx multi-GPU array of float with the x component of the vector quantity to initialise, non-NULL
/// @param Sy multi-GPU array of float with the y component of the vector quantity to initialise, non-NULL
/// @param Sz multi-GPU array of float with the z component of the vector quantity to initialise, non-NULL
/// @param regions multi-GPU array of float with region definition, non-NULL
/// @param initValuesX multi-GPU array of float with initial value of the x component of the vector quantity per region, non-NULL. This array must be ordered by region index value.
/// @param initValuesY multi-GPU array of float with initial value of the y component of the vector quantity per region, non-NULL. This array must be ordered by region index value.
/// @param initValuesZ multi-GPU array of float with initial value of the z component of the vector quantity per region, non-NULL. This array must be ordered by region index value.
/// @note The scalar value must be uniform in eqch region.
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
void initVectorQuantUniformRegionAsync(float** Sx, float** Sy, float** Sz, float** regions, float* initValuesX, float* initValuesY, float* initValuesZ, CUstream* stream, int Npart);


#ifdef __cplusplus
}
#endif
#endif
