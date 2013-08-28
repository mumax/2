/**
  * @file
  * This file implements electrical current paths
  *
  * @author Arne Vansteenkiste
  */

#ifndef _CURRENT_H_
#define _CURRENT_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

// Calculate electrical current density.
// @param jx,jy,jz destination array for result
// @param Ex,Ey,Ez electrical field
// @param rMap space-dependent scalar resistivity (not NULL)
// @param N0,N1Part,N2 array size PER GPU
// @param rMUl multiplier for resistivity
// @param periodic0,periodic1,periodic2 periodic boundary conditions
// @param streams cuda stream PER GPU
DLLEXPORT void currentDensityAsync(float** jx, float** jy, float** jz,
                                   float** Ex, float** Ey, float** Ez,
                                   float** rMap, float rMul,
                                   int N0, int N1Part, int N2,
                                   int periodic0, int periodic1, int periodic2,
                                   CUstream* streams);

DLLEXPORT void diffRhoAsync(float** drho,
                            float** jx, float** jy, float** jz,
                            float cellx, float celly, float cellz,
                            int N0, int N1Part, int N2,
                            int periodic0, int periodic1, int periodic2,
                            CUstream* streams);

#ifdef __cplusplus
}
#endif
#endif
