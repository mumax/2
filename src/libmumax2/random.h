/**
  * @file
  * This file implements the initialisation of a given quantity based on region system.
  * Value will be random in each selected region.
  *
  * @author RŽmy Lassalle-Balier
  */

#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cuda.h>

#ifndef _WIN32
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/// Initialise the scalar quantity S using region system (value is random in each region)
///
/// @param S multi-GPU array of float with the scalar quantity to initialise, non-NULL
/// @param regions multi-GPU array of float with region definition, non-NULL
/// @param host_regionsToProceed multi-GPU array of boolean indicating which region should be proceeded, non-NULL
/// @param regionNum number of regions set in the region system
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
/// @param max (float) maximum value of the randomly generated numbers
/// @param min (float) minimum value of the randomly generated numbers
__declspec(dllexport) void initScalarQuantRandomUniformRegionAsync(float** S,
											 float** regions,
											 bool* host_regionsToProceed,
											 int regionNum,
											 CUstream* stream,
											 int Npart,
											 float max,
											 float min);

/// Initialise the vector quantity S = (Sx,Sy,Sz) using region system (value is random in each region)
///
/// @param Sx multi-GPU array of float with the x component of the vector quantity to initialise, non-NULL
/// @param Sy multi-GPU array of float with the y component of the vector quantity to initialise, non-NULL
/// @param Sz multi-GPU array of float with the z component of the vector quantity to initialise, non-NULL
/// @param regions multi-GPU array of float with region definition, non-NULL
/// @param host_regionsToProceed multi-GPU array of boolean indicating which region should be proceeded, non-NULL
/// @param regionNum number of regions set in the region system
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
__declspec(dllexport) void initVectorQuantRandomUniformRegionAsync(float** Sx, float** Sy, float** Sz,
											 float** regions,
											 bool* host_regionsToProceed,
											 int regionNum,
											 CUstream* stream,
											 int Npart);


#ifdef __cplusplus
}
#endif
#endif
