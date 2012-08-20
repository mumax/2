#include "Q.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16

///@internal
__global__ void Qi3TMDiffKern(float* __restrict__ Qi,
                  const float* __restrict__ Ti, const float* __restrict__ Tj, const float* __restrict__ Tk,
                  const float* __restrict__ lTi, const float* __restrict__ rTi,
                  const float* __restrict__ Q,
                  const float* __restrict__ gamma,
                  const float* __restrict__ Gij, const float* __restrict__ Gik,
                  const float* __restrict__ kMask,
                  const float QMul,
                  const float gammaMul,
                  const float GijMul, const float GikMul,
                  const float kMul,
				  const int isCofT,
                  const int4 size,
				  const float3 mstep,
				  const int3 pbc,
                  int i) {
        
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int k = blockIdx.y * blockDim.y + threadIdx.y;			

        if (j < size.y && k < size.z){ // 3D now:)

            int x0 = i * size.w + j * size.z + k;
                
            int xb1, xf1, x;
            xb1 = (i == 0 && pbc.x == 0) ? i     : i - 1;
            x   = (i == 0 && pbc.x == 0) ? i + 1 : i;
            xf1 = (i == 0 && pbc.x == 0) ? i + 2 : i + 1;
            xb1 = (i == size.x - 1 && pbc.x == 0) ? i - 2 : xb1;
            x   = (i == size.x - 1 && pbc.x == 0) ? i - 1 : x;
            xf1 = (i == size.x - 1 && pbc.x == 0) ? i     : xf1;

            int yb1, yf1, y;       
            yb1 = (j == 0 && lTi == NULL) ? j     : j - 1;
            y   = (j == 0 && lTi == NULL) ? j + 1 : j;
            yf1 = (j == 0 && lTi == NULL) ? j + 2 : j + 1;
            yb1 = (j == size.y - 1 && rTi == NULL) ? j - 2 : yb1;
            y   = (j == size.y - 1 && rTi == NULL) ? j - 1 : y;
            yf1 = (j == size.y - 1 && rTi == NULL) ? j     : yf1; 

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

          
            // Let's use 3-point stencil in the bulk and 3-point forward/backward at the boundary 
            float T_b1, T, T_f1;
            float ddT_x, ddT_y, ddT_z;

            float ddT;
            float sum;
                
            T_b1   = Ti[xn.x];
            T      = Ti[xn.y];
            T_f1   = Ti[xn.z];
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_x = (size.x > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;

            T_b1 = (j > 0 || lTi == NULL) ? Ti[yn.x] : lTi[yn.x];
            T    = Ti[yn.y];
            T_f1 = (j < size.y - 1 || rTi == NULL) ? Ti[yn.z] : rTi[yn.z];   
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_y = (size.y > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;

            T_b1 = Ti[zn.x];
            T    = Ti[zn.y];
            T_f1 = Ti[zn.z]; 
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_z = (size.z > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;
             
            ddT   = mstep.x * ddT_x + mstep.y * ddT_y + mstep.z * ddT_z;
            // ddT is the laplacian(T)
            
            float k = (kMask != NULL) ? kMask[x0] * kMul : kMul;
            float T_i = Ti[x0];
            float T_j = Tj[x0];
            float T_k = Tk[x0];

            float g = (gamma != NULL) ? gamma[x0] * gammaMul : gammaMul;
            float G_ij = (Gij != NULL) ? Gij[x0] * GijMul : GijMul;
            float G_ik = (Gik != NULL) ? Gik[x0] * GikMul : GikMul;

            float C_i = (isCofT == 1) ? g * T_i : g;
            /*if (x0 == 16) {
                printf("Ti: %g\tTj %g\tTk %g\n", T_i, T_j, T_k);
                printf("C_i: %g\tGij %g\tGik %g\n", C_i, G_ij, G_ik);
                printf("k: %g\tddT %g\tQ_i %g\n", k, ddT, Q_i);
            }*/         
            
            C_i = (C_i != 0.0f) ? 1.0f / C_i : 0.0f;
            
            G_ij = G_ij * C_i;
            G_ik = G_ik * C_i;
            k   = k * C_i;
            float Q_i = (Q != NULL) ? Q[x0] * QMul * C_i : QMul * C_i;
            
            Qi[x0] = (G_ij * (T_j - T_i) + G_ik * (T_k - T_i) + Q_i + k * ddT);
        }
}


__global__ void Qi2TMDiffKern(float* __restrict__ Qi,
                  const float* __restrict__ Ti, const float* __restrict__ Tj,
                  const float* __restrict__ lTi, const float* __restrict__ rTi,
                  const float* __restrict__ Q,
                  const float* __restrict__ gamma,
                  const float* __restrict__ Gij,
                  const float* __restrict__ kMask,
                  const float QMul,
                  const float gammaMul,
                  const float GijMul,
                  const float kMul,
				  const int isCofT,
                  const int4 size,
				  const float3 mstep,
				  const int3 pbc,
                  int i) {
        
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int k = blockIdx.y * blockDim.y + threadIdx.y;			

        if (j < size.y && k < size.z){ // 3D now:)

            int x0 = i * size.w + j * size.z + k;
                
            int xb1, xf1, x;
            xb1 = (i == 0 && pbc.x == 0) ? i     : i - 1;
            x   = (i == 0 && pbc.x == 0) ? i + 1 : i;
            xf1 = (i == 0 && pbc.x == 0) ? i + 2 : i + 1;
            xb1 = (i == size.x - 1 && pbc.x == 0) ? i - 2 : xb1;
            x   = (i == size.x - 1 && pbc.x == 0) ? i - 1 : x;
            xf1 = (i == size.x - 1 && pbc.x == 0) ? i     : xf1;

            int yb1, yf1, y;       
            yb1 = (j == 0 && lTi == NULL) ? j     : j - 1;
            y   = (j == 0 && lTi == NULL) ? j + 1 : j;
            yf1 = (j == 0 && lTi == NULL) ? j + 2 : j + 1;
            yb1 = (j == size.y - 1 && rTi == NULL) ? j - 2 : yb1;
            y   = (j == size.y - 1 && rTi == NULL) ? j - 1 : y;
            yf1 = (j == size.y - 1 && rTi == NULL) ? j     : yf1; 

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

          
            // Let's use 3-point stencil in the bulk and 3-point forward/backward at the boundary 
            float T_b1, T, T_f1;
            float ddT_x, ddT_y, ddT_z;

            float ddT;
            float sum;
                
            T_b1   = Ti[xn.x];
            T      = Ti[xn.y];
            T_f1   = Ti[xn.z];
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_x = (size.x > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;

            T_b1 = (j > 0 || lTi == NULL) ? Ti[yn.x] : lTi[yn.x];
            T    = Ti[yn.y];
            T_f1 = (j < size.y - 1 || rTi == NULL) ? Ti[yn.z] : rTi[yn.z];   
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_y = (size.y > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;

            T_b1 = Ti[zn.x];
            T    = Ti[zn.y];
            T_f1 = Ti[zn.z]; 
            sum    = __fadd_rn(T_b1, T_f1);
            ddT_z = (size.z > 2) ? __fmaf_rn(-2.0f, T, sum) : 0.0;
             
            ddT   = mstep.x * ddT_x + mstep.y * ddT_y + mstep.z * ddT_z;
            // ddT is the laplacian(T)
            
            float k = (kMask != NULL) ? kMask[x0] * kMul : kMul;
            float T_i = Ti[x0];
            float T_j = Tj[x0];

            float g = (gamma != NULL) ? gamma[x0] * gammaMul : gammaMul;
            float G_ij = (Gij != NULL) ? Gij[x0] * GijMul : GijMul;

            float C_i = (isCofT == 1) ? g * T_i : g;
            /*if (x0 == 16) {
                printf("Ti: %g\tTj %g\tTk %g\n", T_i, T_j, T_k);
                printf("C_i: %g\tGij %g\tGik %g\n", C_i, G_ij, G_ik);
                printf("k: %g\tddT %g\tQ_i %g\n", k, ddT, Q_i);
            }*/
            
            C_i = (C_i != 0.0f) ? 1.0f / C_i : 0.0f;
            
            G_ij = G_ij * C_i;
            k   = k * C_i;
            float Q_i = (Q != NULL) ? Q[x0] * QMul * C_i : QMul * C_i;
            
            Qi[x0] = (G_ij * (T_j - T_i) + Q_i + k * ddT);
        }
}


__export__ void Q3TM_async(float** Qi,
                         float** Ti, float** Tj, float** Tk, 
                         float** Q, 
                         float** gamma_i, 
                         float** Gij, float** Gik,
                         float** k,
                         float QMul, 
                         float gamma_iMul, 
                         float GijMul, float GikMul, float kMul,
                         int isCofT,
                         const int sx, const int sy, const int sz,
                         const float csx, const float csy, const float csz,
                         const int pbc_x, const int pbc_y, const int pbc_z,
                         CUstream* stream) {
                         
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
		assert(Qi[dev] != NULL);
		assert(Ti[dev] != NULL);
		assert(Tj[dev] != NULL);
		assert(Tk[dev] != NULL);

		gpu_safe(cudaSetDevice(deviceId(dev)));	 
	  		
		// calculate dev neighbours
		
		int ld = Mod(dev - 1, nDev);
		int rd = Mod(dev + 1, nDev);
				
		float* lTi = Ti[ld]; 
		float* rTi = Ti[rd];
		
		if(pbc_y == 0){             
			if(dev == 0){
				lTi = NULL;
			}
			if(dev == nDev-1){
				rTi = NULL;
			}
		}
		
		// printf("Devices are: %d\t%d\t%d\n", ld, dev, rd);
		
		for (int i = 0; i < sx; i++) {
		    Qi3TMDiffKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Qi[dev],
		                                                                Ti[dev], Tj[dev], Tk[dev],
		                                                                lTi, rTi,
		                                                                Q[dev],
		                                                                gamma_i[dev],
		                                                                Gij[dev], Gik[dev],
		                                                                k[dev],
		                                                                QMul,
		                                                                gamma_iMul, 
		                                                                GijMul, GikMul,
		                                                                kMul,
		                                                                isCofT,
		                                                                size,
		                                                                mstep,
		                                                                pbc,i);
        }
	}
}

__export__ void Q2TM_async(float** Qi,
                         float** Ti, float** Tj, 
                         float** Q, 
                         float** gamma_i, 
                         float** Gij,
                         float** k,
                         float QMul, 
                         float gamma_iMul, 
                         float GijMul, float kMul,
                         int isCofT,
                         const int sx, const int sy, const int sz,
                         const float csx, const float csy, const float csz,
                         const int pbc_x, const int pbc_y, const int pbc_z,
                         CUstream* stream) {
                         
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
		assert(Qi[dev] != NULL);
		assert(Ti[dev] != NULL);
		assert(Tj[dev] != NULL);

		gpu_safe(cudaSetDevice(deviceId(dev)));	 
	  		
		// calculate dev neighbours
		
		int ld = Mod(dev - 1, nDev);
		int rd = Mod(dev + 1, nDev);
				
		float* lTi = Ti[ld]; 
		float* rTi = Ti[rd];
		
		if(pbc_y == 0){             
			if(dev == 0){
				lTi = NULL;
			}
			if(dev == nDev-1){
				rTi = NULL;
			}
		}
		
		// printf("Devices are: %d\t%d\t%d\n", ld, dev, rd);
		
		for (int i = 0; i < sx; i++) {
		    Qi2TMDiffKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Qi[dev],
		                                                                Ti[dev], Tj[dev],
		                                                                lTi, rTi,
		                                                                Q[dev],
		                                                                gamma_i[dev],
		                                                                Gij[dev],
		                                                                k[dev],
		                                                                QMul,
		                                                                gamma_iMul, 
		                                                                GijMul,
		                                                                kMul,
		                                                                isCofT,
		                                                                size,
		                                                                mstep,
		                                                                pbc,i);
        }
	}
}


//__export__ void Qs_async(float** Qs,
//                         float** Te, float** Tl, float** Ts, 
//                         float** Cs, 
//                         float** Gsl, float** Ges, 
//                         float** k,
//                         float CsMul, 
//                         float GslMul, float GesMul, float kMul, 
//                         int sx, int sy, sz,
//                         float csx, float csy, float csz,
//                         CUstream* stream, int Npart) {
//    dim3 gridSize, blockSize;
//	make1dconf(Npart, &gridSize, &blockSize);
//	
//	for (int dev = 0; dev < nDevice(); dev++) {
//		assert(Qs[dev] != NULL);
//		assert(Te[dev] != NULL);
//		assert(Tl[dev] != NULL);
//		assert(Ts[dev] != NULL);

//		gpu_safe(cudaSetDevice(deviceId(dev)));
//		QKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>>  (Qs[dev],
//		                                                                Ts[dev], Te[dev], Tl[dev],
//		                                                                Cs[dev],
//		                                                                Ges[dev], Gsl[dev],
//		                                                                k[dev],
//		                                                                CsMul, 
//		                                                                GesMul, GslMul, 
//		                                                                Npart);
//	}
//}
//__export__ void Ql_async(float** Ql,
//                         float** Te, float** Tl, float** Ts, 
//                         float** Cl, 
//                         float** Gel, float** Gsl,
//                         float ClMul,
//                         float GelMul, float GslMul, 
//                         CUstream* stream, int Npart){
//    dim3 gridSize, blockSize;
//	make1dconf(Npart, &gridSize, &blockSize);
//	
//	for (int dev = 0; dev < nDevice(); dev++) {
//		assert(Ql[dev] != NULL);
//		assert(Te[dev] != NULL);
//		assert(Tl[dev] != NULL);
//		assert(Ts[dev] != NULL);
//        
//		gpu_safe(cudaSetDevice(deviceId(dev)));
//		QKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>>  (Ql[dev],
//		                                                                Tl[dev], Te[dev], Ts[dev],
//		                                                                Cl[dev],
//		                                                                Gel[dev], Gsl[dev],
//		                                                                ClMul, 
//		                                                                GelMul, GslMul, 
//		                                                                Npart);
//	}
//}


#ifdef __cplusplus
}
#endif

//__global__ void  QeKern(float* __restrict__ Qi,
//                  const float* __restrict__ Ti, const float* __restrict__ Tj, const float* __restrict__ Tk,
//                  const float* __restrict__ Q,
//                  const float* __restrict__ gamma,
//                  const float* __restrict__ Gij, const float* __restrict__ Gik,
//                  const float QMul,
//                  const float gammaMul,
//                  const float GijMul, const float GikMul,
//                  int NPart) {
//                  
//    int i = threadindex;
//	if (i < NPart) {
//	    float T_i = Ti[i];
//	    float T_j = Tj[i];
//	    float T_k = Tk[i];
//	    
////	    if (i == 10) {
////	        printf(">T_i: %g\tT_j: %g\tT_k: %g\n", T_i, T_j, T_k);
////	    }
//	    
//	    float g = (gamma != NULL) ? gamma[i] * gammaMul : gammaMul;
//	    float Q_i = (Q != NULL) ? Q[i] * QMul : QMul;
//	    float G_ij = (Gij != NULL) ? Gij[i] * GijMul : GijMul;
//	    float G_ik = (Gik != NULL) ? Gik[i] * GikMul : GikMul;
//	    
//	    float C_i = g * T_i;
//	    if (C_i == 0.0f) {
//	        Qi[i] = 0.0;
//	    }
//	    
//	    Qi[i] = (-G_ij * (T_i - T_j) - G_ik * (T_i - T_k) + Q_i) / (g * T_i);
//	}
//}


//__global__  void QKern(float* __restrict__ Qi,
//                  const float* __restrict__ Ti, const float* __restrict__ Tj, const float* __restrict__ Tk,
//                  const float* __restrict__ Ci,
//                  const float* __restrict__ Gij, const float* __restrict__ Gik,
//                  const float CMul,
//                  const float GijMul, const float GikMul,
//                  int NPart) {
//                    
//    int i = threadindex;
//	if (i < NPart) {
//	    float T_i = Ti[i];
//	    float T_j = Tj[i];
//	    float T_k = Tk[i];
//	    
////	    if (i == 10) {
////	        printf(">>T_i: %g\tT_j: %g\tT_k: %g\n", T_i, T_j, T_k);
////	    }
//	    
//	    float C_i = (Ci != NULL) ? Ci[i] * CMul : CMul;
//	    float G_ij = (Gij != NULL) ? Gij[i] * GijMul : GijMul;
//	    float G_ik = (Gik != NULL) ? Gik[i] * GikMul : GikMul;
//	    
//	    if (C_i == 0.0f) {
//	        Qi[i] = 0.0;
//	    }
//	    
//	    Qi[i] = (-G_ij * (T_i - T_j) - G_ik * (T_i - T_k)) / C_i;
//	}
//}

