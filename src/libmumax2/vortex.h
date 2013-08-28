/**
  * @file
  * This file implements the initialisation of a given vectorial quantity based on region system.
  * Value must be either null or vortex in each region.
  *
  * @author RŽmy Lassalle-Balier
  */

#ifndef _VORTEX_H_
#define _VORTEX_H_

#include <cuda.h>
#include "cross_platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the vector quantity S = (Sx,Sy,Sz) as vortex using region system (value is uniform in each region)
/// If a vector null is given for a region then this region will be ignored
///
/// @param Sx multi-GPU array of float with the x component of the vector quantity to initialise, non-NULL
/// @param Sy multi-GPU array of float with the y component of the vector quantity to initialise, non-NULL
/// @param Sz multi-GPU array of float with the z component of the vector quantity to initialise, non-NULL
/// @param regions multi-GPU array of float with region definition, non-NULL
/// @param host_regionsToProceed multi-GPU array of boolean indicating which region should be proceeded, non-NULL
/// @param centerX vortex center X coordinate per region, non-NULL.
/// @param centerY vortex center Y coordinate per region, non-NULL.
/// @param centerZ vortex center Z coordinate per region, non-NULL.
/// @param axisX vortex axis X coordinate per region, non-NULL. The vector must be already normalized.
/// @param axisY vortex axis Y coordinate per region, non-NULL. The vector must be already normalized.
/// @param axisZ vortex axis Z coordinate per region, non-NULL. The vector must be already normalized.
/// @param cellsizeX float standing for the cell size along X direction
/// @param cellsizeY float standing for the cell size along Y direction
/// @param cellsizeZ float standing for the cell size along Z direction
/// @param polarity integer (1 or -1) standing for the polarity of the core
/// @param chirality integer (1 or -1) standing for the chirality of the vortex
/// @param maxRadius float standing for the maximum radius around the core, the vortex should be set. A 0 value means limitless.
/// @param initValNum number of regions set in the region system
/// @param stream multi-GPU streams for asynchronous execution
/// @param Npart number of elements per in each array (i.e. len(mx[0])
/// @param Nx number of elements along x axis
/// @param NyPart number of elements along y axis dealt with on this device (the structure is sliced for multiGPU purpose along Y-axis)
/// @param Nz number of elements along z axis
DLLEXPORT void initVectorQuantVortexRegionAsync(float** Sx, float** Sy, float** Sz,
        float** regions,
        bool* host_regionsToProceed,
        float centerX, float centerY, float centerZ,
        float axisX, float axisY, float axisZ,
        float cellsizeX, float cellsizeY, float cellsizeZ,
        int polarity,
        int chirality,
        float maxRadius,
        int initValNum,
        CUstream* stream,
        int Npart,
        int Nx, int NyPart, int Nz);


#ifdef __cplusplus
}
#endif
#endif
