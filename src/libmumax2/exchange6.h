/**
  * @file
  * This file implements 6-neighbor exchange
  *
  * @todo: implement
  * @author Arne Vansteenkiste
  */

#ifndef _EXCHANGE6_H_
#define _EXCHANGE6_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


void exchange6Async(float** mx, float** my, float** mz,         
                  float** hx, float** hy, float** hz,         
                  int N0, int N1Part, int N2,
                  int peridic0, int periodic1, int periodic2,
                  float cellsizeX, float cellsizeY, float cellsizeZ,
				  CUstream* streams);


#ifdef __cplusplus
}
#endif
#endif
