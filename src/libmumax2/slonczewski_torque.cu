#include "slonczewski_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <cuda.h>
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif
  // ========================================

  __global__ void slonczewski_deltaMKern(float* sttx, float* stty, float* sttz, 
					 float* mx, float* my, float* mz, 
					 float* msat, 
					 float* px, float* py, float* pz,
					 float* jx, float* jy, float* jz,
					 float3 pMul,
					 float3 pre,
					 float3 meshSize,
					 int NPart)
  {
    
    int I = threadindex;
    float Ms = (msat != NULL ) ? msat[I] : 1.0f;
    
	if (Ms > 0.0f && I < NPart){ // Thread configurations are usually too large...

      Ms = 1.0f / Ms;
      pre.y *= Ms;
      pre.z *= Ms;
      
      float3 m = make_float3(mx[I], my[I], mz[I]);
	  
      float p_x = (px != NULL) ? pMul.x * px[I] : pMul.x;
      float p_y = (py != NULL) ? pMul.y * py[I] : pMul.y;
      float p_z = (pz != NULL) ? pMul.z * pz[I] : pMul.z;
    
      float3 p = make_float3(p_x, p_y, p_z);
      float npn = len(p);
      
      float3 pxm = crossf(p, m); // minus
      float3 mxpxm = crossf(pxm, m); // plus
      
      float pdotm = dotf(p,m);
      
      float j_x = (jx != NULL) ? jx[I] : 1.0f / sqrtf(3.0f);
	  float j_y = (jy != NULL) ? jy[I] : 1.0f / sqrtf(3.0f);
	  float j_z = (jz != NULL) ? jz[I] : 1.0f / sqrtf(3.0f);
	  
	  float3 J = make_float3(j_x, j_y, j_z);
	  float nJn = len(J);
	  
	  pre.y *= nJn;
	  pre.z *= nJn;
	  
	  // get effective thinkness of free layer
	  
	  float flt = dotf(meshSize, normalize(J));
	  flt = (flt != 0.0f) ? 1.0f / flt : 1.0f;
	  
      float epsilon = npn * pre.x / ((1.0f + pre.x) + (1.0f - pre.x) * pdotm);
      
      pre.y *= epsilon;

      pre.y *= flt;
      pre.z *= flt;
      
      sttx[I] = pre.y * mxpxm.x + pre.z * pxm.x;
      stty[I] = pre.y * mxpxm.y + pre.z * pxm.y;
      sttz[I] = pre.y * mxpxm.z + pre.z * pxm.z;
      
    } 
  }

  #define BLOCKSIZE 16
  
  void slonczewski_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** msat,
			 float** px, float** py, float** pz,
			 float** jx, float** jy, float** jz,
			 float pxMul, float pyMul, float pzMul,
			 float lambda2, float beta_prime, float pre_field,
			 float meshSizeX,float meshSizeY, float meshSizeZ, 
			 int NPart, 
			 CUstream* stream)
  {

    // 1D configuration
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    float3 meshSize = make_float3(meshSizeX, meshSizeY, meshSizeZ);
    float3 pre = make_float3(lambda2, beta_prime, pre_field);
    float3 pMul = make_float3(pxMul, pyMul, pzMul);
    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
	    slonczewski_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
										       mx[dev], my[dev], mz[dev],
										       msat[dev],  
										       px[dev], py[dev], pz[dev],
										       jx[dev], jy[dev], jz[dev],
											   pMul,
											   pre,
										       meshSize, 
										       NPart);
    } // end dev < nDev loop
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
