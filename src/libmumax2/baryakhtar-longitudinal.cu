#include "baryakhtar-longitudinal.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif  

__global__ void baryakhtarLongitudinalKernFloat(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					
					 float* __restrict__ msat0T0Msk,
					 
					 float* __restrict__ lambda_xx,
					 float* __restrict__ lambda_yy,
					 float* __restrict__ lambda_zz,
					 float* __restrict__ lambda_yz,
					 float* __restrict__ lambda_xz,
					 float* __restrict__ lambda_xy,
					 
					 const float lambdaMul_xx,
					 const float lambdaMul_yy,
					 const float lambdaMul_zz,
					 const float lambdaMul_yz,
					 const float lambdaMul_xz,
					 const float lambdaMul_xy,
				
					 int Npart)
{	
			
		int x0 = threadindex;
		
		if (x0 < Npart){
        
			float msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];
			
			// make sure there is no torque in vacuum!
			if (msat0T0 == 0.0f) {
				tx[x0] = 0.0f;
				ty[x0] = 0.0f;
				tz[x0] = 0.0f;
				return;
			}
		
			float3 H = make_float3(hx[x0], hy[x0], hz[x0]);
		
			float3 lambda_H;
			
			float l_xx = (lambda_xx != NULL) ? lambda_xx[x0] * lambdaMul_xx : lambdaMul_xx;
			float l_xy = (lambda_xy != NULL) ? lambda_xy[x0] * lambdaMul_xy : lambdaMul_xy;
			float l_xz = (lambda_xz != NULL) ? lambda_xz[x0] * lambdaMul_xz : lambdaMul_xz;
			
			lambda_H.x = l_xx * H.x + l_xy * H.y + l_xz * H.z;
			
			float l_yy = (lambda_yy != NULL) ? lambda_yy[x0] * lambdaMul_yy : lambdaMul_yy;
			float l_yz = (lambda_yz != NULL) ? lambda_yz[x0] * lambdaMul_yz : lambdaMul_yz;
		
			lambda_H.y = l_xy * H.x + l_yy * H.y + l_yz * H.z;
		
			float l_zz = (lambda_zz != NULL) ? lambda_zz[x0] * lambdaMul_zz : lambdaMul_zz;
		
			lambda_H.z = l_xz * H.x + l_yz * H.y + l_zz * H.z;
		
			tx[x0] = lambda_H.x;
			ty[x0] = lambda_H.y;
			tz[x0] = lambda_H.z; 
    } 
}

__export__  void baryakhtar_longitudinal_async(float** tx, float**  ty, float**  tz, 
			 float**  hx, float**  hy, float**  hz,
			 
			 float** msat0T0,
			 
			 float** lambda_xx,
			 float** lambda_yy,
			 float** lambda_zz,
			 float** lambda_yz,
			 float** lambda_xz,
			 float** lambda_xy,
			 
			 const float lambdaMul_xx,
			 const float lambdaMul_yy,
			 const float lambdaMul_zz,
			 const float lambdaMul_yz,
			 const float lambdaMul_xz,
			 const float lambdaMul_xy,
			 
			 CUstream* stream,
			 int Npart)
  {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	int nDev = nDevice();
	
	for (int dev = 0; dev < nDev; dev++) {
		
		gpu_safe(cudaSetDevice(deviceId(dev)));	 
		// calculate dev neighbours
		
		baryakhtarLongitudinalKernFloat<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],  
												   hx[dev], hy[dev], hz[dev],
												   
												   msat0T0[dev],
												   
												   lambda_xx[dev],
												   lambda_yy[dev],
												   lambda_zz[dev],
												   lambda_yz[dev],
												   lambda_xz[dev],
												   lambda_xy[dev],

												   lambdaMul_xx,
												   lambdaMul_yy,
												   lambdaMul_zz,
												   lambdaMul_yz,
												   lambdaMul_xz,
												   lambdaMul_xy,
												   
												   Npart);
    } // end dev < nDev loop
	
  }
  
  // ========================================

#ifdef __cplusplus
}
#endif
