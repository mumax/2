/**
  * @file
  * This file implements Slonczewski spin torque
  * See Slonczewski JMMM 159 (1996) L1-L7 and 
  *
  * @author Mykola Dvorni,  Arne Vansteenkiste
  */

#ifndef _LONG_FIELD_H
#define _LONG_FIELD_H

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT    void long_field_async(float** hx, float** hy, float** hz, 
			 float** mx, float** my, float** mz,
			 float** msat, 
			 float** msat0,
			 float** kappa, 
			 float kappaMul,
			 float msatMul, 
			 float msat0Mul,    
			 int NPart, 
			 CUstream* stream);

#ifdef __cplusplus
}
#endif
#endif
