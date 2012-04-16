#include "zhang-li_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

	// dot product
	inline __host__ __device__ float dotf(float3 a, float3 b)
	{ 
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	// cross product
	inline __host__ __device__ float3 crossf(float3 a, float3 b)
	{ 
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
	}
	
  // ========================================

  __global__ void zhangli_deltaMKern(float* sttx, float* stty, float* sttz, 
					 float* mx, float* my, float* mz,					 
					 float* jx, float* jy, float* jz,					 
					 float pred, float pret,
					 int sx, int sy,
					 int sz, int syz,					  	 
					 float i12csx, float i12csy, float i12csz,
					 int i)
  {	
	/*float msat = Msat[I];
	if (msat2 == 0.0f) 
	{
		sttx[I] = 0.0f;
		stty[I] = 0.0f;
		sttz[I] = 0.0f;
		return;
	}*/ 
	
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	
	
    if (j < sy && k < sz){ // 3D now:)
	
	  int x0 = i * syz + j * sz + k;
	  
	  float3 m = make_float3(mx[x0], my[x0], mz[x0]);		
		 	
 
      // First-order derivative 5-points stencil
	   
	  int xb2 = (i-2 >= 0)? i-2 : i;
	  int xb1 = (i-1 >= 0)? i-1 : i;
	  int xf1 = (i+1 < sx)? i+1 : i;
	  int xf2 = (i+2 < sx)? i+2 : i;
	  
	  int yb2 = (j-2 >= 0)? j-2 : j;
	  int yb1 = (j-1 >= 0)? j-1 : j;
	  int yf1 = (j+1 < sy)? j+1 : j;
	  int yf2 = (j+2 < sy)? j+2 : j;
	  
	  int zb2 = (k-2 >= 0)? k-2 : k;
	  int zb1 = (k-1 >= 0)? k-1 : k;
	  int zf1 = (k+1 < sz)? k+1 : k;
	  int zf2 = (k+2 < sz)? k+2 : k;
	  
	  float comm = j * sz + k;
	  int4 xn = make_int4(xb2 * syz + comm, 
						  xb1 * syz + comm, 
						  xf1 * syz + comm, 
						  xf2 * syz + comm); 
						 

	  comm = i * syz + k; 
	  int4 yn = make_int4(yb2 * sz + comm, 
						  yb1 * sz + comm, 
						  yf1 * sz + comm, 
						  yf2 * sz + comm);

	  
	  comm = i * syz + j * sz;
	  int4 zn = make_int4(zb2 + comm, 
						  zb1 + comm, 
						  zf1 + comm, 
						  zf2 + comm);

	  	
	  // Let's use 5-point stencil to avoid problems at the boundaries
	  // CUDA does not have vec3 operators like GLSL has except of .xxx, 
	  // Perhaps for performance need to take into account special cases where j || to x, y or z
	  
	  float3 dmdx = 	make_float3(i12csx * (mx[xn.x] - 8.0f * mx[xn.y] + 8.0f * mx[xn.z] - mx[xn.w]),
									i12csy * (mx[yn.x] - 8.0f * mx[yn.y] + 8.0f * mx[yn.z] - mx[yn.w]),
									i12csz * (mx[zn.x] - 8.0f * mx[zn.y] + 8.0f * mx[zn.z] - mx[zn.w]));
								      
	  float3 dmdy = 	make_float3(i12csx * (my[xn.x] - 8.0f * my[xn.y] + 8.0f * my[xn.z] - my[xn.w]),
								    i12csy * (my[yn.x] - 8.0f * my[yn.y] + 8.0f * my[yn.z] - my[yn.w]),
									i12csz * (my[zn.x] - 8.0f * my[zn.y] + 8.0f * my[zn.z] - my[zn.w]));
										
	  float3 dmdz = 	make_float3(i12csx * (mz[xn.x] - 8.0f * mz[xn.y] + 8.0f * mz[xn.z] - mz[xn.w]),
									i12csy * (mz[yn.x] - 8.0f * mz[yn.y] + 8.0f * mz[yn.z] - mz[yn.w]),
								    i12csz * (mz[zn.x] - 8.0f * mz[zn.y] + 8.0f * mz[zn.z] - mz[zn.w]));  
		
	  // Don't see a point of such overkill, nevertheless:
	  float j_x = 0.0f;
      if(jx != NULL){ j_x += jx[x0]; }
	  
	  float j_y = 0.0f;
      if(jy != NULL){ j_y += jy[x0]; }
	  
	  float j_z = 0.0f;
      if(jz != NULL){ j_z += jz[x0]; }
	  //-------------------------------------------------//
	  		
	  float3 j0 = make_float3(j_x, j_y, j_z);
	  
	  
	  float3 dmdj = make_float3(dotf(dmdx, j0),
								dotf(dmdy, j0),
								dotf(dmdz, j0));
							
	  
	  
	  
      float3 dmdjxm = crossf(dmdj, m); // with minus in it
      
      float3 mxdmxm = crossf(m, dmdjxm); // with minus from [dmdj x m]
	  	  	  
	  sttx[x0] = (pred * mxdmxm.x) + (pret * dmdjxm.x);
      stty[x0] = (pred * mxdmxm.y) + (pret * dmdjxm.y);
      sttz[x0] = (pred * mxdmxm.z) + (pret * dmdjxm.z);   
    } 
  }

  #define BLOCKSIZE 16
  
__export__  void zhangli_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** jx, float** jy, float** jz,
			 const float pred, const float pret,
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 int NPart,
			 CUstream* stream)
  {

	// 3D :)
	
	dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	
	// FUCKING THREADS PER BLOCK LIMITATION
	check3dconf(gridSize, blockSize);
		
	float i12csx = 1.0f / (12.0f * csx);
	float i12csy = 1.0f / (12.0f * csy);
	float i12csz = 1.0f / (12.0f * csz);
		
	int syz = sy * sz;
	
    int nDev = nDevice();
	
	/*cudaEvent_t start,stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);*/
	
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
		// perhaps 16*8*8 block size to get rid of this loop
		for (int i = 0; i < sx; i++) {
			zhangli_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
												   mx[dev], my[dev], mz[dev],											   
												   jx[dev], jy[dev], jz[dev], 
												   pred, pret,
												   sx, sy,
												   sz, syz,										   
												   i12csx, i12csy, i12csz, 
												   i);
		}
    } // end dev < nDev loop
										  
	/*cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Zhang-Li kernel requires: %f ms\n",time);*/
	
  }

  // ========================================

#ifdef __cplusplus
}
#endif
