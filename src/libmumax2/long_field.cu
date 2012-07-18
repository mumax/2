#include "long_field.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <cuda.h>
#include "common_func.h"


#ifdef __cplusplus
extern "C" {
#endif
  // ========================================

  __global__ void long_field_Kern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz, 
					 float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                     float* __restrict__ msatMsk, 
					 float* __restrict__ msat0Msk,
					 float* __restrict__ kappaMsk,
					 float kappaMul,
					 float msatMul,
					 float msat0Mul,
					 int NPart) 
  {
    
    int I = threadindex;
    float Ms0 = (msat0Msk != NULL ) ? msat0Msk[I] * msat0Mul : msat0Mul;
    float kappa = (kappaMsk != NULL ) ? kappaMsk[I] * kappaMul : kappaMul;
    
    if (Ms0 == 0.0f || kappa == 0.0f) {
      hx[I] = 0.0f;
      hy[I] = 0.0f;
      hz[I] = 0.0f;    
      return;
    }
    
    if (I < NPart){ // Thread configurations are usually too large...
      
      kappa = 1.0f / kappa;
      
      float Ms = (msatMsk != NULL ) ? msatMsk[I] * msatMul : msatMul;
      
      float3 m = make_float3(mx[I], my[I], mz[I]);
      
      float ratio = Ms/Ms0;
       
      float mult = Ms * kappa * (1.0f - ratio * ratio);// kappa is actually 0.5/kappa! 
         
      hx[I] = mult * m.x;
      hy[I] = mult * m.y;
      hz[I] = mult * m.z;      
    } 
  }

  #define BLOCKSIZE 16
  
  void long_field_async(float** hx, float** hy, float** hz, 
			 float** mx, float** my, float** mz,
			 float** msat, 
			 float** msat0,
			 float** kappa, 
			 float kappaMul,
			 float msatMul, 
			 float msat0Mul,    
			 int NPart,     
			 CUstream* stream)
  {

    // 1D configuration
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    
    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
	    long_field_Kern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (hx[dev], hy[dev], hz[dev],  
										       mx[dev], my[dev], mz[dev],
										       msat[dev], 
										       msat0[dev],
										       kappa[dev], 
										       kappaMul,
										       msatMul, 
										       msat0Mul,
										       NPart);
    } // end dev < nDev loop
										    
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
