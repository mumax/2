/**
  * @file
  *
  * @author Ben Van de Wiele
  */

#ifndef _DIPOLEKERNEL_H_
#define _DIPOLEKERNEL_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void initFaceKernel6ElementAsync(float **data, 
                                 int co1,
                                 int co2,
                                 int N0, int N1, int N2,          
                                 int N1part,                                /// size of the kernel
                                 int per0, int per1, int per2,              /// periodicity
                                 float cellX, float cellY, float cellZ,     /// cell size
                                 float **dev_qd_P_10, float **dev_qd_W_10,  /// quadrature points and weights
                                 CUstream *streams
                                );

#ifdef __cplusplus
}
#endif
#endif
