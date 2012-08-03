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

__global__ void tbaryakhtar_HdeltaHKernMGPU(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,
					 float* __restrict__ msat0,
					 float* __restrict__ lambda,
					 float* __restrict__ lambda_e,
					 const float msat0Mul,
					 const float lambdaMul,
					 const float lambda_eMul,
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	
    if (j < size.y && k < size.z){ // 3D now:)
        
        int x0 = i * size.w + j * size.z + k;
		    
	    real m_sat = (msat0 != NULL) ? msat0[x0] : 1.0;
	    if (m_sat == 0.0){
	        tx[x0] = 0.0f;
	        ty[x0] = 0.0f;
	        tz[x0] = 0.0f;
	        return;
	    }
	            
	    real l_e = (lambda_e != NULL) ? lambda_e[x0] * lambda_eMul : lambda_eMul;
	    real3 mmstep = make_real3(l_e * mstep.x, l_e * mstep.y, l_e * mstep.z);
	    
	    real pre   = mmstep.z; 
	    real pre_x = mmstep.x / mmstep.z;
	    real pre_y = mmstep.y / mmstep.z;
	    
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
                  
        real h_b1, h, h_f1;
        real ddhx_x, ddhx_y, ddhx_z;
        real ddhy_x, ddhy_y, ddhy_z;
        real ddhz_x, ddhz_y, ddhz_z;
        
        real ddhx, ddhy, ddhz;
        real sum;
        
        h_b1   = hx[xn.x];
        h      = hx[xn.y];
        h_f1   = hx[xn.z];
        sum    = __dadd_rn(h_b1, h_f1);
        ddhx_x = (size.x > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
        
        h_b1 = (j > 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
        h    = hx[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hx[yn.z] : rhx[yn.z];   
        sum    = __dadd_rn(h_b1, h_f1);
        ddhx_y = (size.y > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
    
        h_b1 = hx[zn.x];
        h    = hx[zn.y];
        h_f1 = hx[zn.z]; 
        sum    = __dadd_rn(h_b1, h_f1);
        ddhx_z = (size.z > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
             
        ddhx  = pre * (pre_x * ddhx_x + pre_y * ddhx_y + ddhx_z);
                
        h_b1   = hy[xn.x];
        h      = hy[xn.y];
        h_f1   = hy[xn.z]; 
        sum    = __dadd_rn(h_b1, h_f1);
        ddhy_x = (size.x > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
        
        h_b1 = (j > 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
        h    = hy[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
        sum    = __dadd_rn(h_b1, h_f1);
        ddhy_y = (size.y > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
    
        h_b1 = hy[zn.x];
        h    = hy[zn.y];
        h_f1 = hy[zn.z]; 
        sum    = __dadd_rn(h_b1, h_f1);
        ddhy_z = (size.z > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
         
        ddhy  = pre * (pre_x * ddhy_x + pre_y * ddhy_y + ddhy_z);
			
		h_b1   = hz[xn.x];
        h      = hz[xn.y];
        h_f1   = hz[xn.z]; 
        sum    = __dadd_rn(h_b1, h_f1);
        ddhz_x = (size.x > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
        /*
        if (i == 1 && j == 29 && k > 28 && k < 32) {
            printf("(%d, %d, %d)\thz_y: %.15g %.15g %.15g %.15g\n",i,j,k, h_b1, h, h_f1, ddhz_x);
        }
        */
        h_b1 = (j > 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
        h    = hz[yn.y];
        h_f1 = (j < size.y - 1 || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
        sum    = __dadd_rn(h_b1, h_f1);
        ddhz_y = (size.y > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
        /*
        if (i == 1 && j == 29 && k > 28 && k < 32) {
            printf("(%d, %d, %d)\thz_y: %.15g %.15g %.15g %.15g\n",i,j,k, h_b1, h, h_f1, ddhz_y);
        }
        */
        h_b1 = hz[zn.x];
        h    = hz[zn.y];
        h_f1 = hz[zn.z]; 
        sum    = __dadd_rn(h_b1, h_f1);
        ddhz_z = (size.z > 2) ? __fma_rn(-2.0, h, sum) : 0.0;
         
        ddhz  = pre * (pre_x * ddhz_x + pre_y * ddhz_y + ddhz_z);
        /*
	    if (i == 1 && j == 29 && k > 28 && k < 32) {
            printf("(%d, %d, %d)\thz_y: %.15g %.15g %.15g %.15g %.15g\n",i,j,k, h_b1, h, h_f1, ddhz_z, ddhz);
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
	    float l = (lambda != NULL) ? lambda[x0] * lambdaMul : lambdaMul;
	    float3 H = make_float3(hx[x0], hy[x0], hz[x0]);
        float3 m = make_float3(Mx[x0], My[x0], Mz[x0]);
        
        float3 _mxH = crossf(H, m);

        tx[x0] = _mxH.x + (l * H.x - ddhx);
        ty[x0] = _mxH.y + (l * H.y - ddhy);
        tz[x0] = _mxH.z + (l * H.z - ddhz);  
        
        /*
        tx[x0] = ddhx;
        ty[x0] = ddhy;
        tz[x0] = ddhz;
        */ 
    } 
  }

__global__ void tbaryakhtar_HdeltaHKernMGPUfloat(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,
					 float* __restrict__ msat0,
					 float* __restrict__ lambda,
					 float* __restrict__ lambda_e,
					 const float msat0Mul,
					 const float lambdaMul,
					 const float lambda_eMul,
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	
    if (j < size.y && k < size.z){ // 3D now:)
        
        int x0 = i * size.w + j * size.z + k;
		    
	    float m_sat = (msat0 != NULL) ? msat0[x0] : 1.0;
	    if (m_sat == 0.0){
	        tx[x0] = 0.0f;
	        ty[x0] = 0.0f;
	        tz[x0] = 0.0f;
	        return;
	    }
	            
	    float l_e = (lambda_e != NULL) ? lambda_e[x0] * lambda_eMul : lambda_eMul;
	    float3 mmstep = make_float3(l_e * mstep.x, l_e * mstep.y, l_e * mstep.z);
	    	    
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
	    
	    float l = (lambda != NULL) ? lambda[x0] * lambdaMul : lambdaMul;
	    float3 H = make_float3(hx[x0], hy[x0], hz[x0]);
        float3 m = make_float3(Mx[x0], My[x0], Mz[x0]);
        
        float3 _mxH = crossf(H, m);

        tx[x0] = _mxH.x + (l * H.x - ddhx);
        ty[x0] = _mxH.y + (l * H.y - ddhy);
        tz[x0] = _mxH.z + (l * H.z - ddhz); 
        
        /*
        tx[x0] = ddhx;
        ty[x0] = ddhy;
        tz[x0] = ddhz; 
        */ 
    } 
  }

    
 __global__ void tbaryakhtar_delta2HKernMGPU(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
					 float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
					 float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
					 float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
					 float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,
					 float* __restrict__ msat0,
					 float* __restrict__ lambda,
					 float* __restrict__ lambda_e,
					 const float msat0Mul,
					 const float lambdaMul,
					 const float lambda_eMul,
					 const int4 size,		
					 const float3 mstep,
					 const int3 pbc,
					 const int i)
  {	
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;			
	
    if (j < size.y && k < size.z){ // 3D now:)
        
        int x0 = i * size.w + j * size.z + k;
		    
	    real m_sat = (msat0 != NULL) ? msat0[x0] : 1.0;
	    if (m_sat == 0.0){
	        tx[x0] = 0.0f;
	        ty[x0] = 0.0f;
	        tz[x0] = 0.0f;
	        return;
	    }
	            
	    real l_e = (lambda_e != NULL) ? lambda_e[x0] * lambda_eMul : lambda_eMul;
	    
	    //real one_over_m_sat = 1.0 / m_sat; 
	    /*
	    real5 cfx = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    real5 cfy = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    real5 cfz = make_real5(-1.0, +16.0, -30.0, +16.0, -1.0);
	    */
	    real5 cflb = make_real5(+0.0, +0.0, +1.0, -2.0, +1.0);
        real5 cfrb = make_real5(+1.0, -2.0, +1.0, +0.0, +0.0);
	    
	    real5 cfx = make_real5(+0.0, +12.0, -24.0, +12.0, -0.0);
	    real5 cfy = make_real5(+0.0, +12.0, -24.0, +12.0, -0.0);
	    real5 cfz = make_real5(+0.0, +12.0, -24.0, +12.0, -0.0); 
	    real3 mmstep = make_real3(l_e * mstep.x / 12.0, l_e * mstep.y / 12.0, l_e * mstep.z / 12.0);
 
	    if (pbc.x == 0 && i < 2) {
            cfx.x = 12.0 * cflb.x;
            cfx.y = 12.0 * cflb.y;
            cfx.z = 12.0 * cflb.z;
            cfx.w = 12.0 * cflb.w;
            cfx.v = 12.0 * cflb.v;
        }
        
        if (pbc.x == 0 && i >= size.x - 2) {
            cfx.x = 12.0 * cfrb.x;
            cfx.y = 12.0 * cfrb.y;
            cfx.z = 12.0 * cfrb.z;
            cfx.w = 12.0 * cfrb.w;
            cfx.v = 12.0 * cfrb.v;
        }  
              
        if (lhx == NULL && j < 2) {
            cfy.x = 12.0 * cflb.x;
            cfy.y = 12.0 * cflb.y;
            cfy.z = 12.0 * cflb.z;
            cfy.w = 12.0 * cflb.w;
            cfy.v = 12.0 * cflb.v;
        }
        if (rhx == NULL && j >= size.y - 2) {
            cfy.x = 12.0 * cfrb.x;
            cfy.y = 12.0 * cfrb.y;
            cfy.z = 12.0 * cfrb.z;
            cfy.w = 12.0 * cfrb.w;
            cfy.v = 12.0 * cfrb.v;
        }
        if (pbc.z == 0 && k < 2) {
            cfz.x = 12.0 * cflb.x;
            cfz.y = 12.0 * cflb.y;
            cfz.z = 12.0 * cflb.z;
            cfz.w = 12.0 * cflb.w;
            cfz.v = 12.0 * cflb.v;
        }
        if (pbc.z == 0 && k >= size.z - 2) {
            cfz.x = 12.0 * cfrb.x;
            cfz.y = 12.0 * cfrb.y;
            cfz.z = 12.0 * cfrb.z;
            cfz.w = 12.0 * cfrb.w;
            cfz.v = 12.0 * cfrb.v;
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
         
        HH.x = (yi.x >= 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hy[yn.y] : lhy[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hy[yn.w] : rhy[yn.w];
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
			
		HH.x = (yi.x >= 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
        HH.y = (yi.y >= 0 || lhx == NULL) ? hz[yn.y] : lhz[yn.y];
        HH.z = (yi.z < size.y || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
        HH.w = (yi.w < size.y || rhx == NULL) ? hz[yn.w] : rhz[yn.w];
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
        real ddhz_z = (size.y > 3) ? (cfz.x * h_b2 + cfz.y * h_b1 + cfz.z * H.z + cfz.w * h_f1 + cfz.v * h_f2) : 0.0;  
        real ddhz  = mmstep.x * ddhz_x + mmstep.y * ddhz_y + mmstep.z * ddhz_z;
	           
        real3 le_ddH = make_real3(ddhx, ddhy, ddhz);
        
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
	    real l = (lambda != NULL) ? lambda[x0] * lambdaMul : lambdaMul;
	    
        real3 _mxH = cross(H, m);

        tx[x0] = _mxH.x + (l * H.x - le_ddH.x);
        ty[x0] = _mxH.y + (l * H.y - le_ddH.y);
        tz[x0] = _mxH.z + (l * H.z - le_ddH.z);  
    } 
  }

  
#define BLOCKSIZE 16

__export__  void tbaryakhtar_async(float** tx, float**  ty, float**  tz, 
			 float**  Mx, float**  My, float**  Mz, 
			 float**  hx, float**  hy, float**  hz,
			 float**  msat0,
			 float** lambda,
			 float** lambda_e,
			 const float msat0Mul,
			 const float lambdaMul,
			 const float lambda_eMul,
			 const int sx, const int sy, const int sz,
			 const float csx, const float csy, const float csz,
			 const int pbc_x, const int pbc_y, const int pbc_z, 
			 CUstream* stream)
  {

	// 3D :)
	
	dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
		
	// FUCKING THREADS PER BLOCK LIMITATION
	//check3dconf(gridSize, blockSize);
		
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
			/*									   
			tbaryakhtar_delta2HKernMGPU<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],  
												   Mx[dev], My[dev], Mz[dev],												   
												   hx[dev], hy[dev], hz[dev],
												   lhx, lhy, lhz,
												   rhx, rhy, rhz,
												   msat0[dev],
												   lambda[dev],
												   lambda_e[dev],
												   msat0Mul,
												   lambdaMul,
												   lambda_eMul,
												   size,
												   mstep,
												   pbc,
												   i);
			*/
			tbaryakhtar_HdeltaHKernMGPUfloat<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],  
												   Mx[dev], My[dev], Mz[dev],												   
												   hx[dev], hy[dev], hz[dev],
												   lhx, lhy, lhz,
												   rhx, rhy, rhz,
												   msat0[dev],
												   lambda[dev],
												   lambda_e[dev],
												   msat0Mul,
												   lambdaMul,
												   lambda_eMul,
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
