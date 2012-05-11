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

  __global__ void slonczewski_deltaMKern(float* __restrict__ sttx, float* __restrict__ stty, float* __restrict__ sttz, 
					 float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, 
					 float* __restrict__ msat,
					 float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
					 float* __restrict__ jx, float* __restrict__ jy, float* __restrict__ jz,
					 float* __restrict__ alphaMsk, 
					 float3 pMul,
					 float3 jMul,
					 float3 pre,
					 float3 meshSize,
					 float alphaMul,
					 int NPart) 
  {
    
    int I = threadindex;
    float Ms = (msat != NULL ) ? msat[I] : 1.0f;
    
    if (Ms == 0.0f) {
      sttx[I] = 0.0f;
      stty[I] = 0.0f;
      sttz[I] = 0.0f;    
      return;
    }
    
    float j_x = (jx != NULL) ? jx[I] * jMul.x : jMul.x;
    float j_y = (jy != NULL) ? jy[I] * jMul.y : jMul.y;
    float j_z = (jz != NULL) ? jz[I] * jMul.z : jMul.z;

    float3 J = make_float3(j_x, j_y, j_z);
    float nJn = len(J);
    
    if (nJn == 0.0f) {
      sttx[I] = 0.0f;
      stty[I] = 0.0f;
      sttz[I] = 0.0f;    
      return;  
    }
	  
	if (I < NPart){ // Thread configurations are usually too large...

      Ms = 1.0f / Ms;
          
      pre.y *= Ms;
      pre.z *= Ms;
       
      float3 m = make_float3(mx[I], my[I], mz[I]);
      
      float p_x = (px != NULL) ? pMul.x * px[I] : pMul.x;
      float p_y = (py != NULL) ? pMul.y * py[I] : pMul.y;
      float p_z = (pz != NULL) ? pMul.z * pz[I] : pMul.z;  
        
      float3 p = make_float3(p_x, p_y, p_z);  
                   
      p = normalize(p);
       
      float3 pxm = crossf(p, m); // plus
      float3 mxpxm = crossf(m, pxm); // plus 
      
      float  pdotm = dotf(p, m);
           
	  J = normalize(J);
	  float Jdir = dotf(make_float3(1.0f,1.0f,1.0f), J);
	  float Jsign = Jdir / fabsf(Jdir); 
	  nJn *= Jsign; 
	  pre.y *= nJn;
	  pre.z *= nJn;
	  
	  // get effective thinkness of free layer
	  
	  float free_layer_thickness = fabsf(dotf(meshSize, J)); 
	  free_layer_thickness = (free_layer_thickness != 0.0f) ? 1.0f / free_layer_thickness : 0.0f;
	  pre.y *= free_layer_thickness;
	  pre.z *= free_layer_thickness; 
	  
      float epsilon = pre.x / ((pre.x + 1.0f) + (pre.x - 1.0f) * pdotm);
      pre.y *= epsilon;
      
      float alpha = (alphaMsk != NULL) ? 1.0f/(1.0f + alphaMsk[I] * alphaMul * alphaMsk[I] * alphaMul) : 1.0f/(1.0f + alphaMul * alphaMul); 
      pre.y *= alpha;
      pre.z *= alpha;
     
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
			 float** alphamsk,
			 float pxMul, float pyMul, float pzMul,
			 float jxMul, float jyMul, float jzMul,
			 float lambda2, float beta_prime, float pre_field,
			 float meshSizeX,float meshSizeY, float meshSizeZ, 
			 float alphaMul,
			 int NPart, 
			 CUstream* stream)
  {

    // 1D configuration
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    float3 meshSize = make_float3(meshSizeX, meshSizeY, meshSizeZ);
    float3 pre = make_float3(lambda2, beta_prime, pre_field);
    float3 pMul = make_float3(pxMul, pyMul, pzMul);
    float3 jMul = make_float3(jxMul, jyMul, jzMul);
    
    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
	    slonczewski_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
										       mx[dev], my[dev], mz[dev],
										       msat[dev],  
										       px[dev], py[dev], pz[dev],
										       jx[dev], jy[dev], jz[dev],
										       alphamsk[dev],
											   pMul,
											   jMul,
											   pre,
										       meshSize,
										       alphaMul, 
										       NPart);
    } // end dev < nDev loop
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
