#include "zhang-li_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif
	// mod
	int Mod(int a, int b){
		return (a%b+b)%b;
	}
	
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

  
 __global__ void zhangli_deltaMKernMGPU(float* sttx, float* stty, float* sttz,
					 float* mx, float* my, float* mz,
					 float* lmx, float* lmy, float* lmz,
					 float* rmx, float* rmy, float* rmz,
					 float* jx, float* jy, float* jz,
					 float* msat,
					 float2 pre,
					 int4 size,		
					 float3 mstep,
					 int3 pbc,
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
	   
	  int xb2 = i - 2;
	  int xb1 = i - 1;
	  int xf1 = i + 1;
	  int xf2 = i + 2;
	  
	  int yb2 = j - 2;
	  int yb1 = j - 1;
	  int yf1 = j + 1;
	  int yf2 = j + 2; 
	   
	  int zb2 = k - 2;
	  int zb1 = k - 1;
	  int zf1 = k + 1;
	  int zf2 = k + 2;
	  
	  int4 yi = make_int4(yb2, yb1, yf1, yf2);		  
	  
	  xb2 = (pbc.x == 1 && xb2 < 0)? size.x + xb2 : xb2; // backward coordinates are negative
	  xb1 = (pbc.x == 1 && xb1 < 0)? size.x + xb1 : xb1;
	  xf1 = (pbc.x == 1 && xf1 >= size.x)? xf1 - size.x : xf1;
	  xf2 = (pbc.x == 1 && xf2 >= size.x)? xf2 - size.x : xf2;
	  
	  yb2 = (lmx == NULL && yb2 < 0)? j : yb2;
	  yb1 = (lmx == NULL && yb1 < 0)? j : yb1;
	  yf1 = (rmx == NULL && yf1 > size.y - 1)? j : yf1;
	  yf2 = (rmx == NULL && yf2 > size.y - 1)? j : yf2;
	 	  
	  zb2 = (pbc.z == 1 && zb2 < 0)? (size.z + zb2) : zb2;
	  zb1 = (pbc.z == 1 && zb1 < 0)? (size.z + zb1) : zb1;
	  zf1 = (pbc.z == 1 && zf1 >= size.z)? zf1 - size.z : zf1;
	  zf2 = (pbc.z == 1 && zf2 >= size.z)? zf2 - size.z : zf2;
	 
  	  xb2 = (xb2 >= 0)? xb2 : i;
	  xb1 = (xb1 >= 0)? xb1 : i;
	  xf1 = (xf1 < size.x)? xf1 : i;
	  xf2 = (xf2 < size.x)? xf2 : i;
	  
	  yb2 = (yb2 >= 0)? yb2 : size.y + yb2;
	  yb1 = (yb1 >= 0)? yb1 : size.y + yb1;
	  yf1 = (yf1 < size.y)? yf1 : yf1 - size.y;
	  yf2 = (yf2 < size.y)? yf2 : yf2 - size.y;
	  
		  
      zb2 = (zb2 >= 0)? zb2 : k;
	  zb1 = (zb1 >= 0)? zb1 : k;
	  zf1 = (zf1 < size.z)? zf1 : k;
	  zf2 = (zf2 < size.z)? zf2 : k;	  
	  
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
	  // CUDA does not have vec3 operators like GLSL has, except of .xxx, 
	  // Perhaps for performance need to take into account special cases where j || to x, y or z  
	  
	  float4 MM;
  
	  MM.x = (yi.x >= 0 || lmx == NULL) ? mx[yn.x] : lmx[yn.x];
	  MM.y = (yi.y >= 0 || lmx == NULL) ? mx[yn.y] : lmx[yn.y];
	  MM.z = (yi.z < size.y || rmx == NULL) ? mx[yn.z] : rmx[yn.z];
	  MM.w = (yi.w < size.y || rmx == NULL) ? mx[yn.w] : rmx[yn.w];
	 	  	    
	  float3 dmdx = 	make_float3(mstep.x * (mx[xn.x] - 8.0f * mx[xn.y] + 8.0f * mx[xn.z] - mx[xn.w]),
									mstep.y * (MM.x - 8.0f * MM.y + 8.0f * MM.z - MM.w),
									mstep.z * (mx[zn.x] - 8.0f * mx[zn.y] + 8.0f * mx[zn.z] - mx[zn.w]));
									
      MM.x = (yi.x >= 0 || lmx == NULL) ? my[yn.x] : lmy[yn.x];
	  MM.y = (yi.y >= 0 || lmx == NULL) ? my[yn.y] : lmy[yn.y];
	  MM.z = (yi.z < size.y || rmx == NULL) ? my[yn.z] : rmy[yn.z];
	  MM.w = (yi.w < size.y || rmx == NULL) ? my[yn.w] : rmy[yn.w];
								      
	  float3 dmdy = 	make_float3(mstep.x * (my[xn.x] - 8.0f * my[xn.y] + 8.0f * my[xn.z] - my[xn.w]),
								    mstep.y * (MM.x - 8.0f * MM.y + 8.0f * MM.z - MM.w),
									mstep.z * (my[zn.x] - 8.0f * my[zn.y] + 8.0f * my[zn.z] - my[zn.w]));
									
	  MM.x = (yi.x >= 0 || lmx == NULL) ? mz[yn.x] : lmz[yn.x];
	  MM.y = (yi.y >= 0 || lmx == NULL) ? mz[yn.y] : lmz[yn.y];
	  MM.z = (yi.z < size.y || rmx == NULL) ? mz[yn.z] : rmz[yn.z];
	  MM.w = (yi.w < size.y || rmx == NULL) ? mz[yn.w] : rmz[yn.w]; 								
	  								
										
	  float3 dmdz = 	make_float3(mstep.x * (mz[xn.x] - 8.0f * mz[xn.y] + 8.0f * mz[xn.z] - mz[xn.w]),
									mstep.y * (MM.x - 8.0f * MM.y + 8.0f * MM.z - MM.w),
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
			 const int pbc_x, const int pbc_y, const int pbc_z, 
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
	int3 pbc = make_int3(pbc_x, pbc_y, pbc_z);
	
    int nDev = nDevice();
		
	/*cudaEvent_t start,stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);*/
		
	
	
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));	 
	  		
		// calculate dev neighbours
		
		int ld = Mod(dev - 1, nDev);
		int rd = Mod(dev + 1, nDev);
				
		float* lmx = mx[ld]; 
		float* lmy = my[ld];
		float* lmz = mz[ld];

		float* rmx = mx[rd]; 
		float* rmy = my[rd];
		float* rmz = mz[rd];
		
		if(pbc_y == 0){             
			if(dev == 0){
				lmx = NULL;
				lmy = NULL;
				lmz = NULL;			
			}
			if(dev == nDev-1){
				rmx = NULL;
				rmy = NULL;
				rmz = NULL;
			}
		}
		
		// printf("Devices are: %d\t%d\t%d\n", ld, dev, rd);
		
		for (int i = 0; i < sx; i++) {
			/*zhangli_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
												   mx[dev], my[dev], mz[dev]s,											   
												   jx[dev], jy[dev], jz[dev], 
												   msat[dev],
												   pre,
												   size,
												   mstep,
												   i);*/
												   
			zhangli_deltaMKernMGPU<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
												   mx[dev], my[dev], mz[dev],
												   lmx, lmy, lmz,
												   rmx, rmy, rmz,
												   jx[dev], jy[dev], jz[dev], 
												   msat[dev],												   
												   pre,
												   size,
												   mstep,
												   pbc,
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
