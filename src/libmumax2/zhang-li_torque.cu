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
					 float* msat,
					 float2 pre,
					 int4 size,		
					 float3 mstep,
					 int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	int x0 = i * size.w + j * size.z + k;
	
	float m_sat = (msat != NULL) ? msat[x0] : 1.0f;
	
    if (m_sat != 0.0f && j < size.y && k < size.z){ // 3D now:)
	   
	   m_sat = 1.0f / m_sat;
	  
	  
	  float3 m = make_float3(mx[x0], my[x0], mz[x0]);		
		 	
 
      // First-order derivative 5-points stencil
	   
	  int xb2 = (i-2 >= 0)? i-2 : i;
	  int xb1 = (i-1 >= 0)? i-1 : i;
	  int xf1 = (i+1 < size.x)? i+1 : i;
	  int xf2 = (i+2 < size.x)? i+2 : i;
	  
	  int yb2 = (j-2 >= 0)? j-2 : j;
	  int yb1 = (j-1 >= 0)? j-1 : j;
	  int yf1 = (j+1 < size.y)? j+1 : j;
	  int yf2 = (j+2 < size.y)? j+2 : j;
	  
	  int zb2 = (k-2 >= 0)? k-2 : k;
	  int zb1 = (k-1 >= 0)? k-1 : k;
	  int zf1 = (k+1 < size.z)? k+1 : k;
	  int zf2 = (k+2 < size.z)? k+2 : k;
	  
	  int comm = j * size.z + k;
	  
	  int4 xn = make_int4(xb2 * size.w + comm, 
						  xb1 * size.w + comm, 
						  xf1 * size.w + comm, 
						  xf2 * size.w + comm); 
						 

	  comm = i * size.w + k; 
	  int4 yn = make_int4(yb2 * size.z + comm, 
						  yb1 * size.z + comm, 
						  yf1 * size.z + comm, 
						  yf2 * size.z + comm);

	  
	  comm = i * size.w + j * size.z;
	  int4 zn = make_int4(zb2 + comm, 
						  zb1 + comm, 
						  zf1 + comm, 
						  zf2 + comm);

	  	
	  // Let's use 5-point stencil to avoid problems at the boundaries
	  // CUDA does not have vec3 operators like GLSL has except of .xxx, 
	  // Perhaps for performance need to take into account special cases where j || to x, y or z  
	  
	  float3 dmdx = 	make_float3(mstep.x * (mx[xn.x] - 8.0f * mx[xn.y] + 8.0f * mx[xn.z] - mx[xn.w]),
									mstep.y * (mx[yn.x] - 8.0f * mx[yn.y] + 8.0f * mx[yn.z] - mx[yn.w]),
									mstep.z * (mx[zn.x] - 8.0f * mx[zn.y] + 8.0f * mx[zn.z] - mx[zn.w]));
								      
	  float3 dmdy = 	make_float3(mstep.x * (my[xn.x] - 8.0f * my[xn.y] + 8.0f * my[xn.z] - my[xn.w]),
								    mstep.y * (my[yn.x] - 8.0f * my[yn.y] + 8.0f * my[yn.z] - my[yn.w]),
									mstep.z * (my[zn.x] - 8.0f * my[zn.y] + 8.0f * my[zn.z] - my[zn.w]));
										
	  float3 dmdz = 	make_float3(mstep.x * (mz[xn.x] - 8.0f * mz[xn.y] + 8.0f * mz[xn.z] - mz[xn.w]),
									mstep.y * (mz[yn.x] - 8.0f * mz[yn.y] + 8.0f * mz[yn.z] - mz[yn.w]),
								    mstep.z * (mz[zn.x] - 8.0f * mz[zn.y] + 8.0f * mz[zn.z] - mz[zn.w]));  
		
	  // Don't see a point of such overkill, nevertheless:
	  
	  float3 j0 = make_float3(0.0f, 0.0f, 0.0f);
	  
	  j0.x = (jx != NULL)? jx[x0] : 0.0f; 
	  j0.y = (jy != NULL)? jy[x0] : 0.0f;  
	  j0.z = (jz != NULL)? jz[x0] : 0.0f;
	  
	  //-------------------------------------------------//
	  		  
	  
	  float3 dmdj = make_float3(dotf(dmdx, j0),
								dotf(dmdy, j0),
								dotf(dmdz, j0));
							
	  
	  
	  
      float3 dmdjxm = crossf(dmdj, m); // with minus in it
      
      float3 mxdmxm = crossf(m, dmdjxm); // with minus from [dmdj x m]
	  	  	  
	  sttx[x0] = m_sat*((pre.x * mxdmxm.x) + (pre.y * dmdjxm.x));
      stty[x0] = m_sat*((pre.x * mxdmxm.y) + (pre.y * dmdjxm.y));
      sttz[x0] = m_sat*((pre.x * mxdmxm.z) + (pre.y * dmdjxm.z));   
    } 
  }

  #define BLOCKSIZE 16
  
__export__  void zhangli_async(float** sttx, float** stty, float** sttz, 
			 float** mx, float** my, float** mz, 
			 float** jx, float** jy, float** jz,
			 float** msat,
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
		
	float3 mstep = make_float3(i12csx, i12csy, i12csz);	
	int4 size = make_int4(sx, sy, sz, syz);
	float2 pre = make_float2(pred, pret);
	
    int nDev = nDevice();
	
	/*cudaEvent_t start,stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);*/
	
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));

		for (int i = 0; i < sx; i++) {
			zhangli_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
												   mx[dev], my[dev], mz[dev],											   
												   jx[dev], jy[dev], jz[dev], 
												   msat[dev],
												   pre,
												   size,
												   mstep,
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
