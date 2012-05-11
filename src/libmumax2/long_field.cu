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
					 float kappa,
					 float msatMul,
					 float msat0Mul,
					 int NPart) 
  {
    
    int I = threadindex;
    float ms0 = (msat0Msk != NULL ) ? msat0Msk[I] * msat0Mul : msat0Mul;
    if (ms0 == 0.0f) {
      hx[I] = 0.0f;
      hy[I] = 0.0f;
      hz[I] = 0.0f;    
      return;
    }
    
    if (I < NPart){ // Thread configurations are usually too large...
      float ms = (msatMsk != NULL ) ? msatMsk[I] * msatMul : msatMul;
      
      float3 m = make_float3(mx[I], my[I], mz[I]);
      float3 M = make_float3(0.0f,0.0f,0.0f);
      
      M.x = ms * m.x;
      M.y = ms * m.y;
      M.z = ms * m.z;
      
      float m2 = dotf(m,m);
      float mult = (1.0f - m2 * (ms * ms)/(ms0 * ms0)) * kappa;// kappa is actually 0.5/kappa! 

      float3 h = make_float3(0.0f,0.0f,0.0f);
      
      h.x = mult * M.x;
      h.y = mult * M.y;
      h.z = mult * M.z;
      
      /*if (I == 100) {
        printf("mult: %e\n", mult);
        printf("hx: %e\n", h.x);
        printf("hx: %e\n", h.y);
        printf("hx: %e\n", h.z);
        printf("m2: %e\n", m2);
        printf("ms: %e\n", ms);
        printf("ms0: %e\n", ms0);
      }*/  
      
      hx[I] = h.x;
      hy[I] = h.y;
      hz[I] = h.z;      
    } 
  }

  #define BLOCKSIZE 16
  
  void long_field_async(float** hx, float** hy, float** hz, 
			 float** mx, float** my, float** mz, 
			 float** msat,
			 float** msat0,
			 float kappa,
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
										       kappa, 
										       msatMul,
										       msat0Mul,
										       NPart);
    } // end dev < nDev loop
										    
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
