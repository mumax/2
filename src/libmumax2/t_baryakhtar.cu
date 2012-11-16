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

__global__ void tbaryakhtar_HdeltaHKernMGPUfloat(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,
					 					 
					 float* __restrict__ lambda_xx,
					 float* __restrict__ lambda_yy,
					 float* __restrict__ lambda_zz,
					 float* __restrict__ lambda_yz,
					 float* __restrict__ lambda_xz,
					 float* __restrict__ lambda_xy,
					 
					 float* __restrict__ lambda_e_xx,
					 float* __restrict__ lambda_e_yy,
					 float* __restrict__ lambda_e_zz,
					 float* __restrict__ lambda_e_yz,
					 float* __restrict__ lambda_e_xz,
					 float* __restrict__ lambda_e_xy,
					 
					 const float lambdaMul_xx,
					 const float lambdaMul_yy,
					 const float lambdaMul_zz,
					 const float lambdaMul_yz,
					 const float lambdaMul_xz,
					 const float lambdaMul_xy,
					 
					 const float lambda_eMul_xx,
					 const float lambda_eMul_yy,
					 const float lambda_eMul_zz,
					 const float lambda_eMul_yz,
					 const float lambda_eMul_xz,
					 const float lambda_eMul_xy,
					 
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	
    if (j < size.y && k < size.z){ // 3D now:)
        
        int x0 = i * size.w + j * size.z + k;
	    
	    float3 mmstep = make_float3(mstep.x, mstep.y, mstep.z);
	    	    
        // Second-order derivative 3-points stencil
        int xb1, xf1, x;    
        xb1 = (i == 0 && pbc.x == 0) ? i     : i - 1;
        x   = (i == 0 && pbc.x == 0) ? i + 1 : i;
        xf1 = (i == 0 && pbc.x == 0) ? i + 2 : i + 1;
        xb1 = (i == size.x - 1 && pbc.x == 0) ? i - 2 : xb1;
        x   = (i == size.x - 1 && pbc.x == 0) ? i - 1 : x;
        xf1 = (i == size.x - 1 && pbc.x == 0) ? i     : xf1;
        
        int yb1, yf1, y;       
        yb1 = (j == 0 && lhx == NULL) ? j     : j - 1;
        y   = (j == 0 && lhx == NULL) ? j + 1 : j;
        yf1 = (j == 0 && lhx == NULL) ? j + 2 : j + 1;
        yb1 = (j == size.y - 1 && rhx == NULL) ? j - 2 : yb1;
        y   = (j == size.y - 1 && rhx == NULL) ? j - 1 : y;
        yf1 = (j == size.y - 1 && rhx == NULL) ? j     : yf1; 

        int zb1, zf1, z;       
        zb1 = (k == 0 && pbc.z == 0) ? k     : k - 1;
        z   = (k == 0 && pbc.z == 0) ? k + 1 : k;
        zf1 = (k == 0 && pbc.z == 0) ? k + 2 : k + 1;
        zb1 = (k == size.z - 1 && pbc.z == 0) ? k - 2 : zb1;
        z   = (k == size.z - 1 && pbc.z == 0) ? k - 1 : z;
        zf1 = (k == size.z - 1 && pbc.z == 0) ? k     : zf1; 

        xb1 = (xb1 < 0) ?          size.x + xb1 : xb1;
        xf1 = (xf1 > size.x - 1) ? xf1 - size.x : xf1;    
        
        yb1 = (yb1 < 0) ?          size.y + yb1 : yb1;
        yf1 = (yf1 > size.y - 1) ? yf1 - size.y : yf1;
        
        zb1 = (zb1 < 0) ?          size.z + zb1 : zb1;
        zf1 = (zf1 > size.z - 1) ? zf1 - size.z : zf1; 
        
        /*
        if (i == 0 && j == 127 && k == 127) {
            printf("(%d, %d, %d)\n",xb1,x,xf1);
            printf("(%d, %d, %d)\n",yb1,y,yf1);
            printf("(%d, %d, %d)\n",zb1,z,zf1); 
        }
        
        if (i == 0 && j == 0 && k == 0) {
            printf("(%d, %d, %d)\n",xb1,x,xf1);
            printf("(%d, %d, %d)\n",yb1,y,yf1);
            printf("(%d, %d, %d)\n",zb1,z,zf1); 
        }    
        */
            
        int comm = j * size.z + k;	   
        int3 xn = make_int3(xb1 * size.w + comm,
                            x   * size.w + comm, 
				            xf1 * size.w + comm); 
				         

        comm = i * size.w + k; 
        int3 yn = make_int3(yb1 * size.z + comm,
                            y   * size.z + comm, 
				            yf1 * size.z + comm);


        comm = i * size.w + j * size.z;
        int3 zn = make_int3(zb1 + comm,
                            z   + comm, 
				            zf1 + comm);

          
        // Let's use 5-point stencil in the bulk and 3-point forward/backward at the boundary
                  
        float h_b1, h, h_f1;
        float ddhx_x, ddhx_y, ddhx_z;
        float ddhy_x, ddhy_y, ddhy_z;
        float ddhz_x, ddhz_y, ddhz_z;
        
        float ddhx, ddhy, ddhz;
        float sum;
                
        h_b1   = hx[xn.x];
        h      = hx[xn.y];
        h_f1   = hx[xn.z];
        sum    = __fadd_rn(h_b1, h_f1);
        ddhx_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
        
        h_b1 = (j > 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        h    = hx[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hx[yn.z] : rhx[yn.z];   
        sum    = __fadd_rn(h_b1, h_f1);
        ddhx_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
    
        h_b1 = hx[zn.x];
        h    = hx[zn.y];
        h_f1 = hx[zn.z]; 
        sum    = __fadd_rn(h_b1, h_f1);
        ddhx_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
             
        ddhx   = mmstep.x * ddhx_x + mmstep.y * ddhx_y + mmstep.z * ddhx_z;
                
        h_b1   = hy[xn.x];
        h      = hy[xn.y];
        h_f1   = hy[xn.z]; 
        sum    = __fadd_rn(h_b1, h_f1);
        ddhy_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
        
        h_b1 = (j > 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
        h    = hy[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
        sum    = __fadd_rn(h_b1, h_f1);
        ddhy_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
    
        h_b1 = hy[zn.x];
        h    = hy[zn.y];
        h_f1 = hy[zn.z]; 
        sum    = __fadd_rn(h_b1, h_f1);
        ddhy_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
         
        ddhy   = mmstep.x * ddhy_x + mmstep.y * ddhy_y + mmstep.z * ddhy_z;
			
	h_b1   = hz[xn.x];
        h      = hz[xn.y];
        h_f1   = hz[xn.z]; 
        sum    = __fadd_rn(h_b1, h_f1);
        ddhz_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
        /*
        if (i == 0 && j == 127 && k == 127) {
            printf("(%d, %d, %d)\thz_y: %g %g %g %g\n",i,j,k, h_b1, h, h_f1, ddhz_x);
        }
        */
        
        h_b1 = (j > 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
        h    = hz[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
        sum    = __fadd_rn(h_b1, h_f1);
        ddhz_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
        /*
        if (i == 0 && j == 127 && k == 127) {
            printf("(%d, %d, %d)\thz_y: %g %g %g %g\n",i,j,k, h_b1, h, h_f1, ddhz_y);
        }
        */
        h_b1 = hz[zn.x];
        h    = hz[zn.y];
        h_f1 = hz[zn.z]; 
        sum    = __fadd_rn(h_b1, h_f1);
        ddhz_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;
         
        ddhz   = mmstep.x * ddhz_x + mmstep.y * ddhz_y + mmstep.z * ddhz_z;
        
        /*
	    if (i == 0 && j == 127 && k == 127) {
            printf("(%d, %d, %d)\thz_y: %g %g %g %g %g\n",i,j,k, h_b1, h, h_f1, ddhz_z, ddhz);
        }
        */    
        //real3 le_ddH = make_real3(ddhx, ddhy, ddhz);
        
        /*
	    if (i == 4 && j == 30 && k == 31 ) {
	        printf("(%e, %e, %e)\n",mmstep.x, mmstep.y, mmstep.z);
	        printf("(%e, %e, %e)\n",le_ddH.x, le_ddH.y, le_ddH.z);
	        printf("(%d, %d, %d)\tddhxx: %e\n",i,j,k,ddhx_x);
	        printf("(%d, %d, %d)\tddhxy: %e\n",i,j,k,ddhx_y);
	        printf("(%d, %d, %d)\tddhxz: %e\n",i,j,k,ddhx_z);
	        printf("(%d, %d, %d)\tcfx: %e %e %e %e %e\n",i,j,k,cfx.x, cfx.y, cfx.z, cfx.w, cfx.v);
	        printf("(%d, %d, %d)\tcfy: %e %e %e %e %e\n",i,j,k,cfy.x, cfy.y, cfy.z, cfy.w, cfy.v);
	        printf("(%d, %d, %d)\tcfz: %e %e %e %e %e\n",i,j,k,cfz.x, cfz.y, cfz.z, cfz.w, cfz.v);
	        printf("(%d, %d, %d)\thxx: %e %e %e %e %e\n",i,j,k,hx[xn.x], hx[xn.y], H.x, hx[xn.z], hx[xn.w]);
            printf("(%d, %d, %d)\thxy: %e %e %e %e %e\n",i,j,k,hx[yn.x], hx[yn.y], H.x, hx[yn.z], hx[yn.w]);
            printf("(%d, %d, %d)\thxz: %e %e %e %e %e\n",i,j,k,hx[zn.x], hx[zn.y], H.x, hx[zn.z], hx[zn.w]);
            printf("(%d, %d, %d)\tddhyx: %e\n",i,j,k,ddhy_x);
	        printf("(%d, %d, %d)\tddhyy: %e\n",i,j,k,ddhy_y);
	        printf("(%d, %d, %d)\tddhyz: %e\n",i,j,k,ddhy_z);
            printf("(%d, %d, %d)\thyx: %e %e %e %e %e\n",i,j,k,hy[xn.x], hy[xn.y], H.y, hy[xn.z], hy[xn.w]);
            printf("(%d, %d, %d)\thyy: %e %e %e %e %e\n",i,j,k,hy[yn.x], hy[yn.y], H.y, hy[yn.z], hy[yn.w]);
            printf("(%d, %d, %d)\thyz: %e %e %e %e %e\n",i,j,k,hy[zn.x], hy[zn.y], H.y, hy[zn.z], hy[zn.w]);
            printf("(%d, %d, %d)\tddhyx: %e\n",i,j,k,ddhz_x);
	        printf("(%d, %d, %d)\tddhyy: %e\n",i,j,k,ddhz_y);
	        printf("(%d, %d, %d)\tddhyz: %e\n",i,j,k,ddhz_z);
            printf("(%d, %d, %d)\thzx: %e %e %e %e %e\n",i,j,k,hz[xn.x], hz[xn.y], H.z, hz[xn.z], hz[xn.w]);
            printf("(%d, %d, %d)\thzy: %e %e %e %e %e\n",i,j,k,hz[yn.x], hz[yn.y], H.z, hz[yn.z], hz[yn.w]);
            printf("(%d, %d, %d)\thzz: %e %e %e %e %e\n",i,j,k,hz[zn.x], hz[zn.y], H.z, hz[zn.z], hz[zn.w]); 
	    }
	    */
	    
        float le_xx = (lambda_e_xx != NULL) ? lambda_e_xx[x0] * lambda_eMul_xx : lambda_eMul_xx;
        float le_yy = (lambda_e_yy != NULL) ? lambda_e_yy[x0] * lambda_eMul_yy : lambda_eMul_yy;
        float le_zz = (lambda_e_zz != NULL) ? lambda_e_zz[x0] * lambda_eMul_zz : lambda_eMul_zz;
        float le_yz = (lambda_e_yz != NULL) ? lambda_e_yz[x0] * lambda_eMul_yz : lambda_eMul_yz;
        float le_xz = (lambda_e_xz != NULL) ? lambda_e_xz[x0] * lambda_eMul_xz : lambda_eMul_xz;
        float le_xy = (lambda_e_xy != NULL) ? lambda_e_xy[x0] * lambda_eMul_xy : lambda_eMul_xy;
        
        float ledHx = le_xx * ddhx + le_xy * ddhy + le_xz * ddhz;
        float ledHy = le_xy * ddhx + le_yy * ddhy + le_yz * ddhz;
        float ledHz = le_xz * ddhx + le_yz * ddhy + le_zz * ddhz;
        
        float l_xx = (lambda_xx != NULL) ? lambda_xx[x0] * lambdaMul_xx : lambdaMul_xx;
        float l_yy = (lambda_yy != NULL) ? lambda_yy[x0] * lambdaMul_yy : lambdaMul_yy;
        float l_zz = (lambda_zz != NULL) ? lambda_zz[x0] * lambdaMul_zz : lambdaMul_zz;
        float l_yz = (lambda_yz != NULL) ? lambda_yz[x0] * lambdaMul_yz : lambdaMul_yz;
        float l_xz = (lambda_xz != NULL) ? lambda_xz[x0] * lambdaMul_xz : lambdaMul_xz;
        float l_xy = (lambda_xy != NULL) ? lambda_xy[x0] * lambdaMul_xy : lambdaMul_xy;
        
        float3 H = make_float3(hx[x0], hy[x0], hz[x0]);
        
        float lHx = l_xx * H.x + l_xy * H.y + l_xz * H.z;
        float lHy = l_xy * H.x + l_yy * H.y + l_yz * H.z;
        float lHz = l_xz * H.x + l_yz * H.y + l_zz * H.z;
        
        float3 m = make_float3(Mx[x0], My[x0], Mz[x0]);
        float3 _mxH = crossf(H, m);
        
        tx[x0] = _mxH.x + (lHx - ledHx);
        ty[x0] = _mxH.y + (lHy - ledHy);
        tz[x0] = _mxH.z + (lHz - ledHz); 
    } 
  }

#define BLOCKSIZE 16

__export__  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
			 float**  Mx, float**  My, float**  Mz, 
			 float**  hx, float**  hy, float**  hz,
			 
			 float** lambda_xx,
			 float** lambda_yy,
			 float** lambda_zz,
			 float** lambda_yz,
			 float** lambda_xz,
			 float** lambda_xy,
			 
			 float** lambda_e_xx,
			 float** lambda_e_yy,
			 float** lambda_e_zz,
			 float** lambda_e_yz,
			 float** lambda_e_xz,
			 float** lambda_e_xy,
			 
			 const float lambdaMul_xx,
			 const float lambdaMul_yy,
			 const float lambdaMul_zz,
			 const float lambdaMul_yz,
			 const float lambdaMul_xz,
			 const float lambdaMul_xy,
			 
			 const float lambda_eMul_xx,
			 const float lambda_eMul_yy,
			 const float lambda_eMul_zz,
			 const float lambda_eMul_yz,
			 const float lambda_eMul_xz,
			 const float lambda_eMul_xy,
			 
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 const int pbc_x, const int pbc_y, const int pbc_z, 
			 CUstream* stream)
  {

	// 3D :)
	
	dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
		
	float icsx2 = 1.0f / (csx * csx);
	float icsy2 = 1.0f / (csy * csy);
	float icsz2 = 1.0f / (csz * csz);
	
	int syz = sy * sz;
	
		
	float3 mstep = make_float3(icsx2, icsy2, icsz2);	
	int4 size = make_int4(sx, sy, sz, syz);
	int3 pbc = make_int3(pbc_x, pbc_y, pbc_z);
	
    int nDev = nDevice();		
	
	
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
		
		
		for (int i = 0; i < sx; i++) {
		
			tbaryakhtar_HdeltaHKernMGPUfloat<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],  
												   Mx[dev], My[dev], Mz[dev],												   
												   hx[dev], hy[dev], hz[dev],
												   lhx, lhy, lhz,
												   rhx, rhy, rhz,

												   lambda_xx[dev],
												   lambda_yy[dev],
												   lambda_zz[dev],
												   lambda_yz[dev],
												   lambda_xz[dev],
												   lambda_xy[dev],
												   lambda_e_xx[dev],
												   lambda_e_yy[dev],
												   lambda_e_zz[dev],
												   lambda_e_yz[dev],
												   lambda_e_xz[dev],
												   lambda_e_xy[dev],

												   lambdaMul_xx,
												   lambdaMul_yy,
												   lambdaMul_zz,
												   lambdaMul_yz,
												   lambdaMul_xz,
												   lambdaMul_xy,
												   
												   lambda_eMul_xx,
												   lambda_eMul_yy,
												   lambda_eMul_zz,
												   lambda_eMul_yz,
												   lambda_eMul_xz,
												   lambda_eMul_xy,
												   
												   size,
												   mstep,
												   pbc,
												   i);
		}

    } // end dev < nDev loop
	
  }
  
  // ========================================

#ifdef __cplusplus
}
#endif
