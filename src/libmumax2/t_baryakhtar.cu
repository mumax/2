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
					 float* __restrict__ AexMsk,
					 float* __restrict__ alphaMsk,
					 const float alphaMul,
					 const float pred,
					 const float pre,
					 const float pret,
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	int x0 = i * size.w + j * size.z + k;
		    
	real m_sat = (msat != NULL) ? msat[x0] * msatMul : msatMul;
	
	if (m_sat == 0.0f){
	    tx[x0] = 0.0f;
	    ty[x0] = 0.0f;
	    tz[x0] = 0.0f;
	    
	    l[x0] = 0.0f;
	    return;
	}
	
    if (j < size.y && k < size.z){ // 3D now:)
        
	    m_sat = 1.0f / m_sat;             
        
        real5 cfx = make_real5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    real5 cfy = make_real5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    real5 cfz = make_real5(-1.0f, +16.0f, -30.0f, +16.0f, -1.0f);
	    
	    real3 mmstep = make_real3(mstep.x, mstep.y, mstep.z);
	    
	    if (pbc.x == 0 && i <= 1) {
            cfx.x = +0.0;
            cfx.y = +0.0;
            cfx.z = +1.0;
            cfx.w = -2.0;
            cfx.t = +1.0;
            mmstep.x *= 12.0;
        }
        
        if (pbc.x == 0 && i >= size.x - 2) {
            cfx.x = +1.0;
            cfx.y = -2.0;
            cfx.z = +1.0;
            cfx.w = +0.0;
            cfx.t = +0.0;
            mmstep.x *= 12.0;
        }  
              

        if (pbc.y == 0 && j <= 1) {
            cfy.x = +0.0;
            cfy.y = +0.0;
            cfy.z = +1.0;
            cfy.w = -2.0;
            cfy.t = +1.0;
            mmstep.y *= 12.0;
        }
        if (pbc.y == 0 && j >= size.y - 2) {
            cfy.x = +1.0;
            cfy.y = -2.0;
            cfy.z = +1.0;
            cfy.w = +0.0;
            cfy.t = +0.0;
            mmstep.y *= 12.0;
        }
        if (pbc.z == 0 && k <= 1) {
            cfz.x = +0.0;
            cfz.y = +0.0;
            cfz.z = +1.0;
            cfz.w = -2.0;
            cfz.t = +1.0;
            mmstep.z *= 12.0;
        }
        if (pbc.z == 0 && k >= size.z - 2) {
            cfz.x = +1.0;
            cfz.y = -2.0;
            cfz.z = +1.0;
            cfz.w = +0.0;
            cfz.t = +0.0;
            mmstep.z *= 12.0;
        }
        
        /*if (x0 == 100) {
	        printf("msat: %e  ", m_sat);
	        printf("pre: %e  ", pre);
	        printf("A: %e  ", A);
	        printf("alpha: %e  ", alpha);
	    }*/
	    
        real3 m = make_real3(mx[x0], my[x0], mz[x0]);		
        
        // Longitudinal part
        
        real3 h = make_real3(hx[x0], hy[x0], hz[x0]);
        
        real lr = lambda * dot(h, m); // lambda * (H, m)   
        
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


        // Let's use 5-point stencil to avoid problems at the boundaries
        // CUDA does not have vec3 operators like GLSL has, except of .xxx, 
        // Perhaps for performance need to take into account special cases where j || to x, y or z  

        real4 HH;

        HH.x = (yi.x >= 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hx[yn.y] : lhx[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hx[yn.z] : rhx[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hx[yn.w] : rhx[yn.w];
          	    
        real3 dhxdr2 = 	make_real3(mmstep.x * (cfx.x * hx[xn.x] + cfx.y * hx[xn.y] + cfx.z * hx[x0] + cfx.w * hx[xn.z] + cfx.t * hx[xn.w]),
							            mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hx[x0] + cfy.w * HH.z     + cfy.t * HH.w),
							            mmstep.z * (cfz.x * hx[zn.x] + cfz.y * hx[zn.y] + cfz.z * hx[x0] + cfz.w * hx[zn.z] + cfz.t * hx[zn.w]));
							
        HH.x = (yi.x >= 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hy[yn.y] : lhy[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hy[yn.w] : rhy[yn.w];
						              
        real3 dhydr2 = 	make_real3(mmstep.x * (cfx.x * hy[xn.x] + cfx.y * hy[xn.y] + cfx.z * hy[x0] + cfx.w * hy[xn.z] + cfx.t * hy[xn.w]),
						                mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hy[x0] + cfy.w * HH.z     + cfy.t * HH.w),
							            mmstep.z * (cfz.x * hy[zn.x] + cfz.y * hy[zn.y] + cfz.z * hy[x0] + cfz.w * hy[zn.z] + cfz.t * hy[zn.w]));
							
        HH.x = (yi.x >= 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hz[yn.y] : lhz[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hz[yn.w] : rhz[yn.w]; 								
							
								
        real3 dhzdr2 = 	make_real3(mmstep.x * (cfx.x * hz[xn.x] + cfx.y * hz[xn.y] + cfx.z * hz[x0] + cfx.w * hz[xn.z] + cfx.t * hz[xn.w]),
							            mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hz[x0] + cfy.w * HH.z     + cfy.t * HH.w),
						                mmstep.z * (cfz.x * hz[zn.x] + cfz.y * hz[zn.y] + cfz.z * hz[x0] + cfz.w * hz[zn.z] + cfz.t * hz[zn.w])); 


	            
        real3 ddh = make_real3(dhxdr2.x + dhxdr2.y + dhxdr2.z, dhydr2.x + dhydr2.y + dhydr2.z, dhzdr2.x + dhzdr2.y + dhzdr2.z);

		// Longitudinal part
					
	    real le = lambda_e * dot(m, ddh); // Lambda_e * (m, laplace(h)  
	    l[x0] = (lr - le) / msatMul; // lr - le, since normalize m/As to 1/s, gammaLL is in multiplier
	    /*if (x0 == 0) {
	        printf("lr: %e\n",lr);
	        printf("le: %e\n",le);
	        printf("lr-le: %e\n",(lr-le)/msatMul);
	        printf("m: %e\n", msatMul);
	    }*/
 	    //*****************    	  
	    
        real3 ddhxm = cross(m, ddh); // no minus in it, but it was an interesting behaviour when damping is pumping

        real3 mxddhxm = cross(m, ddhxm); // with plus from [ddh x m]    
        
        real3 _mxh = cross(h, m);
        real3 _mxmxh = cross(m, _mxh);
        
        tx[x0] = _mxh.x + m_sat * (lambda * _mxmxh.x  + lambda_e * mxddhxm.x);
        ty[x0] = _mxh.y + m_sat * (lambda * _mxmxh.y  + lambda_e * mxddhxm.y);
        tz[x0] = _mxh.z + m_sat * (lambda * _mxmxh.z  + lambda_e * mxddhxm.z);  
        
        /*if (x0 == 0) {
            printf("mx: %e\n",m.x);
	        printf("my: %e\n",m.y);
	        printf("mz: %e\n",m.z);
	        
	        printf("hx: %e\n",h.x);
	        printf("hy: %e\n",h.y);
	        printf("hz: %e\n",h.z);
	        
	        printf("mxhx: %e\n",_mxh.x);
	        printf("mxhy: %e\n",_mxh.y);
	        printf("mxhz: %e\n",_mxh.z);
	        
	        printf("mxmxhx: %e\n",_mxmxh.x);
	        printf("mxmxhy: %e\n",_mxmxh.y);
	        printf("mxmxhz: %e\n",_mxmxh.z);
	    }*/

    } 
  }

  
#define BLOCKSIZE 16


  
__export__  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
             float** l,
			 float**  mx, float**  my, float**  mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat,
			 float**  AexMsk,
			 float**  alphaMsk,
			 const float alphaMul,
			 const float pred,
			 const float pre,
			 const float pret,
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
												   AexMsk[dev],							 
												   alphaMsk[dev],
												   alphaMul,
												   pred,
												   pre,
												   pret,
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
