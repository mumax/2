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
        
        
        real5 cfx = make_real5(-1.0 / 12.0, +16.0 / 12.0, -30.0 / 12.0, +16.0 / 12.0, -1.0 / 12.0);
	    real5 cfy = make_real5(-1.0 / 12.0, +16.0 / 12.0, -30.0 / 12.0, +16.0 / 12.0, -1.0 / 12.0);
	    real5 cfz = make_real5(-1.0 / 12.0, +16.0 / 12.0, -30.0 / 12.0, +16.0 / 12.0, -1.0 / 12.0);
	    
	    
	    real5 cflb = make_real5(+0.0, +0.0, +1.0, -2.0, +1.0);
        real5 cfrb = make_real5(+1.0, -2.0, +1.0, +0.0, +0.0);
	    
	    /*
	    real5 cfx = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0);
	    real5 cfy = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0);
	    real5 cfz = make_real5(+0.0, +1.0, -2.0, +1.0, -0.0); 
	    */
	    real3 mmstep = make_real3(lambda_e * mstep.x, lambda_e * mstep.y, lambda_e * mstep.z);
	    
	    if (pbc.x == 0 && i < 2) {
            cfx.x = cflb.x;
            cfx.y = cflb.y;
            cfx.z = cflb.z;
            cfx.w = cflb.w;
            cfx.v = cflb.v;
        }
        
        if (pbc.x == 0 && i >= size.x - 2) {
            cfx.x = cfrb.x;
            cfx.y = cfrb.y;
            cfx.z = cfrb.z;
            cfx.w = cfrb.w;
            cfx.v = cfrb.v;
        }  
              
        if (lhx == NULL && j < 2) {
            cfy.x = cflb.x;
            cfy.y = cflb.y;
            cfy.z = cflb.z;
            cfy.w = cflb.w;
            cfy.v = cflb.v;
        }
        if (rhx == NULL && j >= size.y - 2) {
            cfy.x = cfrb.x;
            cfy.y = cfrb.y;
            cfy.z = cfrb.z;
            cfy.w = cfrb.w;
            cfy.v = cfrb.v;
        }
        if (pbc.z == 0 && k < 2) {
            cfz.x = cflb.x;
            cfz.y = cflb.y;
            cfz.z = cflb.z;
            cfz.w = cflb.w;
            cfz.v = cflb.v;
        }
        if (pbc.z == 0 && k >= size.z - 2) {
            cfz.x = cfrb.x;
            cfz.y = cfrb.y;
            cfz.z = cfrb.z;
            cfz.w = cfrb.w;
            cfz.v = cfrb.v;
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
        real3 H = make_real3(hx[x0], hy[x0], hz[x0]);
           
        real4 HH;
     
        HH.x = (yi.x >= 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hx[yn.y] : lhx[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hx[yn.z] : rhx[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hx[yn.w] : rhx[yn.w];
        
        real h_b2 = hx[xn.x];
        real h_b1 = hx[xn.y];
        real h_f1 = hx[xn.z];
        real h_f2 = hx[xn.w]; 
        real ddhx_x = (size.x > 3) ? (cfx.x * h_b2 + cfx.y * h_b1 + cfx.z * H.x + cfx.w * h_f1 + cfx.v * h_f2) : 0.0;
        real ddhx_y = (size.y > 3) ? (cfy.x * HH.x + cfy.y * HH.y + cfy.z * H.x + cfy.w * HH.z + cfy.v * HH.w) : 0.0;
        
        h_b2 = hx[zn.x];
        h_b1 = hx[zn.y];
        h_f1 = hx[zn.z];
        h_f2 = hx[zn.w];
        real ddhx_z = (size.z > 3) ? (cfz.x * h_b2 + cfz.y * h_b1 + cfz.z * H.x + cfz.w * h_f1 + cfz.v * h_f2) : 0.0; 
        
        real ddhx  = mmstep.x * ddhx_x + mmstep.y * ddhx_y + mmstep.z * ddhx_z;
        
        
        
        HH.x = (yi.x >= 0 || lhy == NULL) ? hy[yn.x] : lhy[yn.x];
        HH.y = (yi.y >= 0 || lhy == NULL) ? hy[yn.y] : lhy[yn.y];
        HH.z = (yi.z < size.y || rhy == NULL) ? hy[yn.z] : rhy[yn.z];
        HH.w = (yi.w < size.y || rhy == NULL) ? hy[yn.w] : rhy[yn.w];
        
        h_b2 = hy[xn.x];
        h_b1 = hy[xn.y];
        h_f1 = hy[xn.z];
        h_f2 = hy[xn.w]; 
        real ddhy_x = (size.x > 3) ? (cfx.x * h_b2 + cfx.y * h_b1 + cfx.z * H.y + cfx.w * h_f1 + cfx.v * h_f2) : 0.0;
        real ddhy_y = (size.y > 3) ? (cfy.x * HH.x + cfy.y * HH.y + cfy.z * H.y + cfy.w * HH.z + cfy.v * HH.w) : 0.0;
        
        h_b2 = hy[zn.x];
        h_b1 = hy[zn.y];
        h_f1 = hy[zn.z];
        h_f2 = hy[zn.w];
        real ddhy_z = (size.z > 3) ? (cfz.x * h_b2 + cfz.y * h_b1 + cfz.z * H.y + cfz.w * h_f1 + cfz.v * h_f2) : 0.0;  
        
        real ddhy  = mmstep.x * ddhy_x + mmstep.y * ddhy_y + mmstep.z * ddhy_z;
		
        h_b2 = hz[xn.x];
        h_b1 = hz[xn.y];
        h_f1 = hz[xn.z];
        h_f2 = hz[xn.w]; 
        real ddhz_x = (size.x > 3) ? (cfx.x * h_b2 + cfx.y * h_b1 + cfx.z * H.z + cfx.w * h_f1 + cfx.v * h_f2) : 0.0;
        real ddhz_y = (size.y > 3) ? (cfy.x * HH.x + cfy.y * HH.y + cfy.z * H.z + cfy.w * HH.z + cfy.v * HH.w) : 0.0;
        
        h_b2 = hz[zn.x];
        h_b1 = hz[zn.y];
        h_f1 = hz[zn.z];
        h_f2 = hz[zn.w];
        real ddhz_z = (size.z > 3) ? (cfz.x * h_b2 + cfz.y * h_b1 + cfz.z * H.z + cfz.w * h_f1 + cfz.v * h_f2) : 0.0;  
        
        real ddhz  = mmstep.x * ddhz_x + mmstep.y * ddhz_y + mmstep.z * ddhz_z;
	            
        real3 le_ddH = make_real3(ddhx, ddhy, ddhz);
        
    
	    /*if (i == 2 && j == 2 && k == 2) {
	        printf("(%d, %d, %d)\tddhxx: %e\n",i,j,k,ddhx_x);
	        printf("(%d, %d, %d)\tddhxy: %e\n",i,j,k,ddhx_y);
	        printf("(%d, %d, %d)\tddhxz: %e\n",i,j,k,ddhx_z);
	        printf("(%d, %d, %d)\tcfx: %e %e %e %e %e\n",i,j,k,cfx.x, cfx.y, cfx.z, cfx.w, cfx.v);
	        printf("(%d, %d, %d)\tcfy: %e %e %e %e %e\n",i,j,k,cfy.x, cfy.y, cfy.z, cfy.w, cfy.v);
	        printf("(%d, %d, %d)\tcfz: %e %e %e %e %e\n",i,j,k,cfz.x, cfz.y, cfz.z, cfz.w, cfz.v);
	        printf("(%d, %d, %d)\thxx: %e %e %e %e %e\n",i,j,k,hx[xn.x], hx[xn.y], H.x, hx[xn.z], hx[xn.w]);
            printf("(%d, %d, %d)\thxy: %e %e %e %e %e\n",i,j,k,hx[yn.x], hx[yn.y], H.x, hx[yn.z], hx[yn.w]);
            printf("(%d, %d, %d)\thxz: %e %e %e %e %e\n",i,j,k,hx[zn.x], hx[zn.y], H.x, hx[zn.z], hx[zn.w]);
            printf("(%d, %d, %d)\tddhxx: %e\n",i,j,k,ddhy_x);
	        printf("(%d, %d, %d)\tddhxy: %e\n",i,j,k,ddhy_y);
	        printf("(%d, %d, %d)\tddhxz: %e\n",i,j,k,ddhy_z);
            printf("(%d, %d, %d)\thxx: %e %e %e %e %e\n",i,j,k,hy[xn.x], hy[xn.y], H.y, hy[xn.z], hy[xn.w]);
            printf("(%d, %d, %d)\thxy: %e %e %e %e %e\n",i,j,k,hy[yn.x], hy[yn.y], H.y, hy[yn.z], hy[yn.w]);
            printf("(%d, %d, %d)\thxz: %e %e %e %e %e\n",i,j,k,hy[zn.x], hy[zn.y], H.y, hy[zn.z], hy[zn.w]);
            printf("(%d, %d, %d)\tddhxx: %e\n",i,j,k,ddhz_x);
	        printf("(%d, %d, %d)\tddhxy: %e\n",i,j,k,ddhz_y);
	        printf("(%d, %d, %d)\tddhxz: %e\n",i,j,k,ddhz_z);
            printf("(%d, %d, %d)\thxx: %e %e %e %e %e\n",i,j,k,hz[xn.x], hz[xn.y], H.z, hz[xn.z], hz[xn.w]);
            printf("(%d, %d, %d)\thxy: %e %e %e %e %e\n",i,j,k,hz[yn.x], hz[yn.y], H.z, hz[yn.z], hz[yn.w]);
            printf("(%d, %d, %d)\thxz: %e %e %e %e %e\n",i,j,k,hz[zn.x], hz[zn.y], H.z, hz[zn.z], hz[zn.w]); 
	    }*/
	    
	    real nmn = len(m);
	    
        real3 _mxH = cross(H, m);
                    
        tx[x0] = _mxH.x + nmn * (lambda * H.x - le_ddH.x);
        ty[x0] = _mxH.y + nmn * (lambda * H.y - le_ddH.y);
        tz[x0] = _mxH.z + nmn * (lambda * H.z - le_ddH.z);  

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
		
	float icsx2 = 1.0f / (csx * csx);
	float icsy2 = 1.0f / (csy * csy);
	float icsz2 = 1.0f / (csz * csz);
	
	int syz = sy * sz;
	
		
	float3 mstep = make_float3(icsx2, icsy2, icsz2);	
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
