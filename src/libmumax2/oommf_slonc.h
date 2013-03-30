/**
  * @file
  * This file implements OOMMF style Slonczewski spin torque
  * See Slonczewski JMMM 159 (1996) L1-L7 and 
  *
  * @author Mykola Dvorni, Graham Rowlands, Arne Vansteenkiste
  */

#ifndef _OOMMF_SLONC_H_
#define _OOMMF_SLONC_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

  DLLEXPORT void oommf_slonc_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** msat,
			 float** px, float** py, float** pz,
			 float** jx, float** jy, float** jz,
			 float** alphamsk,
			 float** t_flmsk,
			 float** polFreMsk,float** polFixMsk,
			 float** lambdaFreMsk,float** lambdaFixMsk,
			 float** epsilon_primeMsk,
			 float pxMul, float pyMul, float pzMul,
			 float jxMul, float jyMul, float jzMul,
			 float beta, float pre_field,
			 float meshSizeX,float meshSizeY, float meshSizeZ, 
			 float alphaMul,
			 float t_flMul,
			 float polFreMul,float polFixMul,
			 float lambdaFreMul,float lambdaFixMul,
			 int NPart, 
			 CUstream* stream);

#ifdef __cplusplus
}
#endif
#endif
