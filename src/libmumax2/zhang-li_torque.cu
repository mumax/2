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
					 float i12csx_2, float i12csy_2, float i12csz_2,
					 int NPart)
  {
		
	int I = threadindex;
				
    if (I < NPart){ // Thread configurations are usually too large...
	  
	  float3 m = make_float3(mx[I], my[I], mz[I]);
	
	  float msat2 = dotf(m, m);
		
		if (msat2 == 0.0f) 
		{
			sttx[I] = 0.0f;
			stty[I] = 0.0f;
			sttz[I] = 0.0f;
			return;
		}  
	  	
		int k = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int i = blockIdx.z * blockDim.z + threadIdx.z;	
		
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

	  int x0 = comm + k;
	  	
	  // Let's use 5-point stencil to avoid problems at the boundaries
	  
	  float3 dmx = 	make_float3(i12csx_2 * (-mx[xn.x] + 8.0f * mx[xn.y] - 8.0f * mx[xn.z] + mx[xn.w]),
								i12csx_2 * (-my[xn.x] + 8.0f * my[xn.y] - 8.0f * my[xn.z] + my[xn.w]),
								i12csx_2 * (-mz[xn.x] + 8.0f * my[xn.y] - 8.0f * mz[xn.z] + mz[xn.w]));
	  
	  float3 dmy = 	make_float3(i12csy_2 * (-mx[yn.x] + 8.0f * mx[yn.y] - 8.0f * mx[yn.z] + mx[yn.w]),
								i12csy_2 * (-my[yn.x] + 8.0f * my[yn.y] - 8.0f * my[yn.z] + my[yn.w]),
								i12csy_2 * (-mz[yn.x] + 8.0f * my[yn.y] - 8.0f * mz[yn.z] + mz[yn.w]));
										
	  float3 dmz = 	make_float3(i12csz_2 * (-mx[zn.x] + 8.0f * mx[zn.y] - 8.0f * mx[zn.z] + mx[zn.w]),
								i12csz_2 * (-my[zn.x] + 8.0f * my[zn.y] - 8.0f * my[zn.z] + my[zn.w]),
								i12csz_2 * (-mz[zn.x] + 8.0f * my[zn.y] - 8.0f * mz[zn.z] + mz[zn.w]));
	  
	  
		
	  // Don't see a point of such overkill, nevertheless:
	  float j_x = 0.0;
      if(jx != NULL){ j_x += jx[x0]; }
	  
	  float j_y = 0.0;
      if(jy != NULL){ j_y += jy[x0]; }
	  
	  float j_z = 0.0;
      if(jz != NULL){ j_z += jz[x0]; }
	  //-------------------------------------------------//
	  		
	  float3 j0 = make_float3(j_x, j_y, j_z);

	  float3 dm = make_float3(dotf(dmx, j0),
								dotf(dmy, j0),
								dotf(dmz, j0));
							
	  
	  
	  
      float3 dmxm = crossf(dm, m);
      
      float3 mxdmxm = crossf(m, dmxm);
	  	  
	  float jj = sqrtf(dotf(j0, j0));
	  
	  float at = jj * pret;
	  float ad = jj * pred;
	  
      sttx[x0] = ((ad * mxdmxm.x) + (at * dmxm.x));
      stty[x0] = ((ad * mxdmxm.y) + (at * dmxm.y));
      sttz[x0] = ((ad * mxdmxm.z) + (at * dmxm.z));
	       
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

    // 1D configuration
    //dim3 gridSize, blockSize;
    //make1dconf(NPart, &gridSize, &blockSize);
	// 3D instead
	
	dim3 gridSize(divUp(sz, BLOCKSIZE), divUp(sy, BLOCKSIZE), divUp(sx, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
	printf("[ZLSTT DEBUG] The grid size: %d x %d x %d\n", sz, sy, sx);
	
	float i12csx_2 = 1.0f / (12.0f * csx * csz);
	float i12csy_2 = 1.0f / (12.0f * csy * csy);
	float i12csz_2 = 1.0f / (12.0f * csz * csz);
	
	printf("[ZLSTT DEBUG] Step Prefactors are: %f & %f & %f\n", i12csx_2, i12csy_2, i12csz_2);
	printf("[ZLSTT DEBUG] Prefactors are: %f & %f\n", pred, pret);
	
	int syz = sy * sz;
	
    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++) {
      gpu_safe(cudaSetDevice(deviceId(dev)));
	    zhangli_deltaMKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (sttx[dev], stty[dev], sttz[dev],  
											   mx[dev], my[dev], mz[dev],											   
											   jx[dev], jy[dev], jz[dev], 
											   pred, pret,
											   sx, sy,
											   sz, syz,										   
											   i12csx_2, i12csy_2, i12csz_2, 
											   NPart);
    } // end dev < nDev loop
										  
										  
  }

  // ========================================

#ifdef __cplusplus
}
#endif
