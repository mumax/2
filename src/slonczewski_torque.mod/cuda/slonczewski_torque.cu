#include "slonczewski_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif
  // ========================================

  __global__ void slonczewski_deltaMKern(float* sttx, float* stty, float* sttz, 
					 float* mx, float* my, float* mz, 
					 float* px, float* py, float* pz,
					 float* alpha, float* Msat,
					 float gamma, float aj, float bj, float Pol, 
					 float *curr, 
					 int NPart)
  {
    
    int I = threadindex;
	if (I < NPart){ // Thread configurations are usually too large...

      //float Ms = Msat[I];
    
    //if (Ms > 0.0) { // do bother if there's nothing here, branching makes cuda code slower.
      float m_x = mx[I];
      float m_y = my[I];
      float m_z = mz[I];
      float p_x = px[I];
      float p_y = py[I];
      float p_z = pz[I];

      float pxm_x = -p_y * m_z + m_y * p_z;
      float pxm_y =  p_x * m_z - m_x * p_z;
      float pxm_z = -p_x * m_y + m_x * p_y;
      
      float mxpxm_x = -pxm_y * m_z + m_y * pxm_z;
      float mxpxm_y =  pxm_x * m_z - m_x * pxm_z;
      float mxpxm_z = -pxm_x * m_y + m_x * pxm_y;

      sttx[I] = mxpxm_x;//0.0*mxpxm_x;
      stty[I] = mxpxm_y;//0.0*mxpxm_y;
      sttz[I] = mxpxm_z;//0.0*mxpxm_z;
      
    } 
        
  }

  #define BLOCKSIZE 16
  
  void slonczewski_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** px, float** py, float** pz,
			 float** alpha, float** Msat,
			 float gamma, float aj, float bj, float Pol,
			 float **curr, 
			 int NPart, 
			 CUstream* stream)
  {

    // 1D configuration
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);

    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
	    slonczewski_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
										       mx[dev], my[dev], mz[dev],  
										       px[dev], py[dev], pz[dev],
										       alpha[dev], Msat[dev], gamma, aj, bj, Pol, curr[dev], 
										       NPart);
    } // end dev < nDev loop
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
