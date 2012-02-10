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
					 int N0, int N1Part, int N2,
					 int i) 
  {
    
    //  i is the device index, x coordinate
    int j = blockIdx.x * blockDim.x + threadIdx.x; // y coordinate
    int k = blockIdx.y * blockDim.y + threadIdx.y; // z coordinate
    int I = i*N1Part*N2 + j*N2 + k; // linear array index

    float Ms = Msat[I];
    
    if (Ms > 0.0) { // don't bother if there's nothing here!
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

      sttx[I] = 0.0*mxpxm_x;
      stty[I] = 0.0*mxpxm_y;
      sttz[I] = 0.0*mxpxm_z;
      
    } // end if (Msat > 0.0)
        
  }

  #define BLOCKSIZE 16
  
  void slonczewski_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** px, float** py, float** pz,
			 float** alpha, float** Msat,
			 float gamma, float aj, float bj, float Pol,
			 float **curr, 
			 int N0, int N1Part, int N2, 
			 CUstream* stream)
  {
    dim3 gridSize(divUp(N1Part, BLOCKSIZE), divUp(N2, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
      for (int i = 0; i < N0; i++) {
	slonczewski_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
										       mx[dev], my[dev], mz[dev],  
										       px[dev], py[dev], pz[dev],
										       alpha[dev], Msat[dev], gamma, aj, bj, Pol, curr[dev], 
										       N0, N1Part, N2, i);
      } // end i < N0 loop
    } // end dev < nDev loop
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
