#include "t_baryakhtar.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif
   
 __global__ void tbaryakhtar_delta2HKernMGPU(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                     float* __restrict__ l,
					 float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,	 
					 float* __restrict__ msat,
					 const float msatMul,
					 const float lambda,
					 const float lambda_e,
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	int x0 = i * size.w + j * size.z + k;
		    
	float m_sat = (msat != NULL) ? msat[x0] * msatMul : msatMul;
	
	if (m_sat == 0.0f){
	    tx[x0] = 0.0f;
	    ty[x0] = 0.0f;
	    tz[x0] = 0.0f;
	    
	    l[x0] = 0.0f;
	    return;
	}
	
    if (j < size.y && k < size.z){ // 3D now:)
        
	    m_sat = 1.0f / m_sat;             
        
        float5 cfx = make_float5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    float5 cfy = make_float5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    float5 cfz = make_float5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    
	    float3 mmstep = mstep;
	    
	    if (pbc.x == 0 && i <= 1) {
            cfx.x = +0.0f;
            cfx.y = +0.0f;
            cfx.z = +1.0f;
            cfx.w = -2.0f;
            cfx.t = +1.0f;
            mmstep.x *= 12.0f;
        }
        
        if (pbc.x == 0 && i >= size.x - 2) {
            cfx.x = +1.0f;
            cfx.y = -2.0f;
            cfx.z = +1.0f;
            cfx.w = +0.0f;
            cfx.t = +0.0f;
            mmstep.x *= 12.0f;
        }  
              

        if (pbc.y == 0 && j <= 1) {
            cfy.x = +0.0f;
            cfy.y = +0.0f;
            cfy.z = +1.0f;
            cfy.w = -2.0f;
            cfy.t = +1.0f;
            mmstep.y *= 12.0f;
        }
        if (pbc.y == 0 && j >= size.y - 2) {
            cfy.x = +1.0f;
            cfy.y = -2.0f;
            cfy.z = +1.0f;
            cfy.w = +0.0f;
            cfy.t = +0.0f;
            mmstep.y *= 12.0f;
        }
        if (pbc.z == 0 && k <= 1) {
            cfz.x = +0.0f;
            cfz.y = +0.0f;
            cfz.z = +1.0f;
            cfz.w = -2.0f;
            cfz.t = +1.0f;
            mmstep.z *= 12.0f;
        }
        if (pbc.z == 0 && k >= size.z - 2) {
            cfz.x = +1.0f;
            cfz.y = -2.0f;
            cfz.z = +1.0f;
            cfz.w = +0.0f;
            cfz.t = +0.0f;
            mmstep.z *= 12.0f;
        }
        
        /*if (x0 == 100) {
	        printf("msat: %e  ", m_sat);
	        printf("pre: %e  ", pre);
	        printf("A: %e  ", A);
	        printf("alpha: %e  \n", alpha);
	        printf("prel: %e  ", prel);
	    }*/
	    
        float3 m = make_float3(mx[x0], my[x0], mz[x0]);		
        
        // Longitudinal part
        
        float3 h = make_float3(hx[x0], hy[x0], hz[x0]);
        
        float lr = lambda * dotf(h, m); // lambda * (H, m)   
        
        // Transverse part    
         
        
        // Second-order derivative 5-points stencil

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

        xb2 = (pbc.x == 0 && xb2 < 0)? i : xb2; // backward coordinates are negative
        xb1 = (pbc.x == 0 && xb1 < 0)? i : xb1;
        xf1 = (pbc.x == 0 && xf1 >= size.x)? i : xf1;
        xf2 = (pbc.x == 0 && xf2 >= size.x)? i : xf2;
        
        /*if (i == 0)
        {   
           printf("cfx: %e %e %e %e %e\n", cfx.x, cfx.y, cfx.z, cfx.w, cfx.t);
        }*/
        
        yb2 = (lhx == NULL && yb2 < 0)? j : yb2;
        yb1 = (lhx == NULL && yb1 < 0)? j : yb1;
        yf1 = (rhx == NULL && yf1 >= size.y)? j : yf1;
        yf2 = (rhx == NULL && yf2 >= size.y)? j : yf2;

        zb2 = (pbc.z == 0 && zb2 < 0)? k : zb2;
        zb1 = (pbc.z == 0 && zb1 < 0)? k : zb1;
        zf1 = (pbc.z == 0 && zf1 >= size.z)? k : zf1;
        zf2 = (pbc.z == 0 && zf2 >= size.z)? k : zf2;
                
        xb2 = (xb2 >= 0)? xb2 : size.x + xb2;
        xb1 = (xb1 >= 0)? xb1 : size.x + xb1;
        xf1 = (xf1 < size.x)? xf1 : xf1 - size.x;
        xf2 = (xf2 < size.x)? xf2 : xf2 - size.x;

        yb2 = (yb2 >= 0)? yb2 : size.y + yb2;
        yb1 = (yb1 >= 0)? yb1 : size.y + yb1;
        yf1 = (yf1 < size.y)? yf1 : yf1 - size.y;
        yf2 = (yf2 < size.y)? yf2 : yf2 - size.y;

        zb2 = (zb2 >= 0)? zb2 : size.z + zb2;
        zb1 = (zb1 >= 0)? zb1 : size.z + zb1;
        zf1 = (zf1 < size.z)? zf1 : zf1 - size.z;
        zf2 = (zf2 < size.z)? zf2 : zf2 - size.z;
          
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


        // Let's use 5-point stencil in the bulk and 3-point forward/backward at the boundary
        // CUDA does not have vec3 operators like GLSL has, except of .xxx, 

        float4 HH;

        HH.x = (yi.x >= 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hx[yn.y] : lhx[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hx[yn.z] : rhx[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hx[yn.w] : rhx[yn.w];
          	    
        float3 dhxdr2 = 	make_float3(mmstep.x * (cfx.x * hx[xn.x] + cfx.y * hx[xn.y] + cfx.z * hx[x0] + cfx.w * hx[xn.z] + cfx.t * hx[xn.w]),
							            mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hx[x0] + cfy.w * HH.z     + cfy.t * HH.w),
							            mmstep.z * (cfz.x * hx[zn.x] + cfz.y * hx[zn.y] + cfz.z * hx[x0] + cfz.w * hx[zn.z] + cfz.t * hx[zn.w]));
							
        HH.x = (yi.x >= 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hy[yn.y] : lhy[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hy[yn.w] : rhy[yn.w];
						              
        float3 dhydr2 = 	make_float3(mmstep.x * (cfx.x * hy[xn.x] + cfx.y * hy[xn.y] + cfx.z * hy[x0] + cfx.w * hy[xn.z] + cfx.t * hy[xn.w]),
						                mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hy[x0] + cfy.w * HH.z     + cfy.t * HH.w),
							            mmstep.z * (cfz.x * hy[zn.x] + cfz.y * hy[zn.y] + cfz.z * hy[x0] + cfz.w * hy[zn.z] + cfz.t * hy[zn.w]));
							
        HH.x = (yi.x >= 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hz[yn.y] : lhz[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hz[yn.w] : rhz[yn.w]; 								
							
								
        float3 dhzdr2 = 	make_float3(mmstep.x * (cfx.x * hz[xn.x] + cfx.y * hz[xn.y] + cfx.z * hz[x0] + cfx.w * hz[xn.z] + cfx.t * hz[xn.w]),
							            mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hz[x0] + cfy.w * HH.z     + cfy.t * HH.w),
						                mmstep.z * (cfz.x * hz[zn.x] + cfz.y * hz[zn.y] + cfz.z * hz[x0] + cfz.w * hz[zn.z] + cfz.t * hz[zn.w])); 


	            
        float3 ddh = make_float3(dhxdr2.x + dhxdr2.y + dhxdr2.z, dhydr2.x + dhydr2.y + dhydr2.z, dhzdr2.x + dhzdr2.y + dhzdr2.z);

		// Longitudinal part
					
	    float le = lambda_e * dotf(m, ddh); // Lambda_e * (m, laplace(h)  
	    l[x0] = (lr - le) / msatMul; // lr - le, since normalize m/As to 1/s, gammaLL is in multiplier
	    
	    //*****************    	  
	    
        float3 ddhxm = crossf(m, ddh); // no minus in it, but it was an interesting behaviour when damping is pumping

        float3 mxddhxm = crossf(m, ddhxm); // with plus from [ddh x m]    
        
        float3 _mxh = crossf(h, m);
        float3 _mxmxh = crossf(m, _mxh);
        
        tx[x0] = _mxh.x + m_sat * (lambda * _mxmxh.x  + lambda_e * mxddhxm.x);
        ty[x0] = _mxh.y + m_sat * (lambda * _mxmxh.y  + lambda_e * mxddhxm.y);
        tz[x0] = _mxh.z + m_sat * (lambda * _mxmxh.z  + lambda_e * mxddhxm.z);  
        

    } 
  }

  
#define BLOCKSIZE 16


  
__export__  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
             float** l,
			 float**  mx, float**  my, float**  mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat,
			 const float msatMul, 
			 const float lambda,
			 const float lambda_e,
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
		
	float i12csx = 1.0f / (12.0f * csx * csx);
	float i12csy = 1.0f / (12.0f * csy * csy);
	float i12csz = 1.0f / (12.0f * csz * csz);
	
	int syz = sy * sz;
	
		
	float3 mstep = make_float3(i12csx, i12csy, i12csz);	
	int4 size = make_int4(sx, sy, sz, syz);
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
				
		float* lhx = hx[ld]; 
		float* lhy = hy[ld];
		float* lhz = hz[ld];

		float* rhx = hx[rd]; 
		float* rhy = hy[rd];
		float* rhz = hz[rd];
		
		if(pbc_y == 0){             
			if(dev == 0){
				lhx = NULL;
				lhy = NULL;
				lhz = NULL;			
			}
			if(dev == nDev-1){
				rhx = NULL;
				rhy = NULL;
				rhz = NULL;
			}
		}
		
		// printf("Devices are: %d\t%d\t%d\n", ld, dev, rd);
		
		for (int i = 0; i < sx; i++) {
												   
			tbaryakhtar_delta2HKernMGPU<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],  
			                                       l[dev],
												   mx[dev], my[dev], mz[dev],												   
												   hx[dev], hy[dev], hz[dev],
												   lhx, lhy, lhz,
												   rhx, rhy, rhz,
												   msat[dev],
												   msatMul,
												   lambda,
												   lambda_e,
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
	printf("T-Baryakhtar kernel requires: %f ms\n",time);*/
	
  }
  
  // ========================================

#ifdef __cplusplus
}
#endif
