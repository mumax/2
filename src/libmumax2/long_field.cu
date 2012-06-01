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
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz, 
					 float* __restrict__ msat0Msk,
					 float kappa,
					 float msat0Mul,
					 int NPart) 
  {
    
    int I = threadindex;
    real ms0 = (msat0Msk != NULL ) ? msat0Msk[I] * msat0Mul : msat0Mul;
    
    if (ms0 == 0.0f) {
      hx[I] = 0.0f;
      hy[I] = 0.0f;
      hz[I] = 0.0f;    
      return;
    }
    
    if (I < NPart){ // Thread configurations are usually too large...
    
      real3 M = make_real3(Mx[I], My[I], Mz[I]);
        
      real MM = dot(M,M);
      real MMS = ms0 * ms0;
      
      real mult = kappa * (1.0f - (MM/MMS));// kappa is actually 0.5/kappa! 
             
      hx[I] = mult * M.x;
      hy[I] = mult * M.y;
      hz[I] = mult * M.z;      
    } 
  }

  #define BLOCKSIZE 16
  
  void long_field_async(float** hx, float** hy, float** hz, 
			 float** Mx, float** My, float** Mz, 
			 float** msat0, 
			 float kappa,
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
										       Mx[dev], My[dev], Mz[dev], 
										       msat0[dev], 
										       kappa, 
										       msat0Mul,
										       NPart);
    } // end dev < nDev loop
										    
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
