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
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,
					 float* __restrict__ msat0,
					 const float msat0Mul,
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
		    
	real m_sat = (msat0 != NULL) ? msat0[x0] : 1.0;
	
	if (m_sat == 0.0){
	    tx[x0] = 0.0f;
	    ty[x0] = 0.0f;
	    tz[x0] = 0.0f;
	    return;
	}
	
    if (j < size.y && k < size.z){ // 3D now:)
            
	    //m_sat = 1.0 / m_sat;             
        
        real5 cfx = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    real5 cfy = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    real5 cfz = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    
	    /*real5 cfx = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0);
	    real5 cfy = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0);
	    real5 cfz = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0);*/
	    
	    
	    real3 mmstep = make_real3(mstep.x, mstep.y, mstep.z);
	    
	    if (pbc.x == 0 && i <= 1) {
            cfx.x = +0.0;
            cfx.y = +0.0;
            cfx.z = +1.0;
            cfx.w = -2.0;
            cfx.v = +1.0;
            mmstep.x *= 12.0;
        }
        
        if (pbc.x == 0 && i >= size.x - 2) {
            cfx.x = +1.0;
            cfx.y = -2.0;
            cfx.z = +1.0;
            cfx.w = +0.0;
            cfx.v = +0.0;
            mmstep.x *= 12.0;
        }  
              

        if (pbc.y == 0 && j <= 1) {
            cfy.x = +0.0;
            cfy.y = +0.0;
            cfy.z = +1.0;
            cfy.w = -2.0;
            cfy.v = +1.0;
            mmstep.y *= 12.0;
        }
        if (pbc.y == 0 && j >= size.y - 2) {
            cfy.x = +1.0;
            cfy.y = -2.0;
            cfy.z = +1.0;
            cfy.w = +0.0;
            cfy.v = +0.0;
            mmstep.y *= 12.0;
        }
        if (pbc.z == 0 && k <= 1) {
            cfz.x = +0.0;
            cfz.y = +0.0;
            cfz.z = +1.0;
            cfz.w = -2.0;
            cfz.v = +1.0;
            mmstep.z *= 12.0;
        }
        if (pbc.z == 0 && k >= size.z - 2) {
            cfz.x = +1.0;
            cfz.y = -2.0;
            cfz.z = +1.0;
            cfz.w = +0.0;
            cfz.v = +0.0;
            mmstep.z *= 12.0;

        }
     
        real3 m = make_real3(Mx[x0], My[x0], Mz[x0]);		
          
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

        
        real4 HH;

        HH.x = (yi.x >= 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hx[yn.y] : lhx[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hx[yn.z] : rhx[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hx[yn.w] : rhx[yn.w];
                	    
        real ddhx  =  mmstep.x * (cfx.x * hx[xn.x] + cfx.y * hx[xn.y] + cfx.z * hx[x0] + cfx.w * hx[xn.z] + cfx.v * hx[xn.w])
			        + mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hx[x0] + cfy.w * HH.z     + cfy.v * HH.w)
			        + mmstep.z * (cfz.x * hx[zn.x] + cfz.y * hx[zn.y] + cfz.z * hx[x0] + cfz.w * hx[zn.z] + cfz.v * hx[zn.w]);
							
        HH.x = (yi.x >= 0 || lhy == NULL) ? hy[yn.x] : lhy[yn.x];
        HH.y = (yi.y >= 0 || lhy == NULL) ? hy[yn.y] : lhy[yn.y];
        HH.z = (yi.z < size.y || rhy == NULL) ? hy[yn.z] : rhy[yn.z];
        HH.w = (yi.w < size.y || rhy == NULL) ? hy[yn.w] : rhy[yn.w];
		        				              
        real ddhy  =  mmstep.x * (cfx.x * hy[xn.x] + cfx.y * hy[xn.y] + cfx.z * hy[x0] + cfx.w * hy[xn.z] + cfx.v * hy[xn.w])
					+ mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hy[x0] + cfy.w * HH.z     + cfy.v * HH.w)
					+ mmstep.z * (cfz.x * hy[zn.x] + cfz.y * hy[zn.y] + cfz.z * hy[x0] + cfz.w * hy[zn.z] + cfz.v * hy[zn.w]);
							
        HH.x = (yi.x >= 0 || lhz == NULL) ? hz[yn.x] : lhz[yn.x];
        HH.y = (yi.y >= 0 || lhz == NULL) ? hz[yn.y] : lhz[yn.y];
        HH.z = (yi.z < size.y || rhz == NULL) ? hz[yn.z] : rhz[yn.z];
        HH.w = (yi.w < size.y || rhz == NULL) ? hz[yn.w] : rhz[yn.w]; 								
		            				
								
        real ddhz  =   mmstep.x * (cfx.x * hz[xn.x] + cfx.y * hz[xn.y] + cfx.z * hz[x0] + cfx.w * hz[xn.z] + cfx.v * hz[xn.w])
					 + mmstep.y * (cfy.x * HH.x     + cfy.y * HH.y     + cfy.z * hz[x0] + cfy.w * HH.z     + cfy.v * HH.w)
	                 + mmstep.z * (cfz.x * hz[zn.x] + cfz.y * hz[zn.y] + cfz.z * hz[x0] + cfz.w * hz[zn.z] + cfz.v * hz[zn.w]); 

	            
        real3 ddH = make_real3(ddhx, ddhy, ddhz);
        real3 H = make_real3(hx[x0], hy[x0], hz[x0]);
        
	    /*if (i==0 && j == 0 && k == 0) {
	        printf("ddh.x: %e\n",ddh.x);
	        printf("ddh.y: %e\n",ddh.y);
	        printf("ddh.z: %e\n",ddh.z);
	        printf("hx_xb2: %e\n",hx[xn.x]);
	        printf("hx_xb1: %e\n",hx[xn.y]);
	        printf("hx_x: %e\n",  hx[x0]);
	        printf("hx_xf1: %e\n",hx[xn.z]);
	        printf("hx_xf2: %e\n",hx[xn.w]);
	        
	        printf("hx_yb2: %e\n",hx[yn.x]);
	        printf("hx_yb1: %e\n",hx[yn.y]);
	        printf("hx_y: %e\n",  hx[x0]);
	        printf("hx_yf1: %e\n",hx[yn.z]);
	        printf("hx_yf2: %e\n",hx[yn.w]);
	        
	        printf("hx_zb2: %e\n",hx[zn.x]);
	        printf("hx_zb1: %e\n",hx[zn.y]);
	        printf("hx_z: %e\n",  hx[x0]);
	        printf("hx_zf1: %e\n",hx[zn.z]);
	        printf("hx_zf2: %e\n",hx[zn.w]);
	        
	        printf("hy_xb2: %e\n",hy[xn.x]);
	        printf("hy_xb1: %e\n",hy[xn.y]);
	        printf("hy_x: %e\n",  hy[x0]);
	        printf("hy_xf1: %e\n",hy[xn.z]);
	        printf("hy_xf2: %e\n",hy[xn.w]);
	        
	        printf("hy_yb2: %e\n",hy[yn.x]);
	        printf("hy_yb1: %e\n",hy[yn.y]);
	        printf("hy_y: %e\n",  hy[x0]);
	        printf("hy_yf1: %e\n",hy[yn.z]);
	        printf("hy_yf2: %e\n",hy[yn.w]);
	        
	        printf("hy_zb2: %e\n",hy[zn.x]);
	        printf("hy_zb1: %e\n",hy[zn.y]);
	        printf("hy_z: %e\n",  hy[x0]);
	        printf("hy_zf1: %e\n",hy[zn.z]);
	        printf("hy_zf2: %e\n",hy[zn.w]);
	        
	        printf("hz_xb2: %e\n",hz[xn.x]);
	        printf("hz_xb1: %e\n",hz[xn.y]);
	        printf("hz_x: %e\n",  hz[x0]);
	        printf("hz_xf1: %e\n",hz[xn.z]);
	        printf("hz_xf2: %e\n",hz[xn.w]);
	        
	        printf("hz_yb2: %e\n",hz[yn.x]);
	        printf("hz_yb1: %e\n",hz[yn.y]);
	        printf("hz_y: %e\n",  hz[x0]);
	        printf("hz_yf1: %e\n",hz[yn.z]);
	        printf("hz_yf2: %e\n",hz[yn.w]);
	        
	        printf("hz_zb2: %e\n",hz[zn.x]);
	        printf("hz_zb1: %e\n",hz[zn.y]);
	        printf("hz_z: %e\n",  hz[x0]);
	        printf("hz_zf1: %e\n",hz[zn.z]);
	        printf("hz_zf2: %e\n",hz[zn.w]);
	        
	        
	        
	    }*/
	          
        real3 _mxH = cross(H, m);
        
        /*if (x0 == 100) {
            printf("Ms: %e\n",Msat);
            printf("lambda: %e\n",lambda);
            printf("lambda_e: %e\n",lambda_e);
            printf("Mx: %e\n",M.x);
            printf("My: %e\n",M.y);
            printf("Mz: %e\n",M.z);
            
        } */            
        
        tx[x0] = _mxH.x + (lambda * H.x  - lambda_e * ddH.x);
        ty[x0] = _mxH.y + (lambda * H.y  - lambda_e * ddH.y);
        tz[x0] = _mxH.z + (lambda * H.z  - lambda_e * ddH.z);  

    } 
  }

  
#define BLOCKSIZE 16


  
__export__  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
			 float**  Mx, float**  My, float**  Mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat0,
			 const float msat0Mul,
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
												   Mx[dev], My[dev], Mz[dev],												   
												   hx[dev], hy[dev], hz[dev],
												   lhx, lhy, lhz,
												   rhx, rhy, rhz,
												   msat0[dev],
												   msat0Mul,
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
