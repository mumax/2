#include "slonczewski_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif
  // ========================================

  __global__ void slonczewski_deltaMKern(float *mx, float* my, float* mz, 
					 float* hx, float* hy, float* hz,
					 float* px, float* py, float* pz,
					 float* alpha, float* Msat,
					 float bj, float cj, float *curr,
					 float dt_gilb,
					 int N0, int N1Part, int N2,
					 int i) 
  {
    
    //  i is passed
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int I = i*N1Part*N2 + j*N2 + k; // linear array index

    int blah = i+j+I;
    blah = blah*blah;
    
    float pxm_x = -py * mz + my * pz;
    float pxm_y =  px * mz - mx * pz;
    float pxm_z = -px * my + mx * py;
    
    float mxpxm_x = -pxm_y * mz + my * pxm_z;
    float mxpxm_y =  pxm_x * mz - mx * pxm_z;
    float mxpxm_z = -pxm_x * my + mx * pxm_y;
        
  }

  void slonczewski_deltaMAsync(float** mx, float** my, float** mz, 
			       float** hx, float** hy, float** hz,
			       float** px, float** py, float** pz,
			       float** alpha, float** Msat,
			       float bj, float cj, float **curr,
			       float dt_gilb,
			       CUstream* stream,
			       int Npart) 
  {
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    slonczewski_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (mx, my, mz, hx, hy, hz,px, py, pz,
										   alpha, Msat, bj, cj, curr,dt_gilb,Npart);
      
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
