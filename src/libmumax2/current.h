/**
  * @file
  * This file implements electrical current paths
  *
  * @author Arne Vansteenkiste
  */

#ifndef _CURRENT_H_
#define _CURRENT_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void currentDensityAsync(float** jx, float** jy, float** jz,
                         float** Ex, float** Ey, float** Ez, 
                         float** rMap, float rMul, 
                         int N0, int N1Part, int N2,
                         int periodic0, int periodic1, int periodic2, 
                         float cellx, float celly, float cellz, 
                         CUstream* streams);


#ifdef __cplusplus
}
#endif
#endif
