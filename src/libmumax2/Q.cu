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

///@internal

__global__ void  QeKern(float* __restrict__ Qi,
                  const float* __restrict__ Ti, const float* __restrict__ Tj, const float* __restrict__ Tk,
                  const float* __restrict__ Q,
                  const float* __restrict__ gamma,
                  const float* __restrict__ Gij, const float* __restrict__ Gik,
                  const float QMul,
                  const float gammaMul,
                  const float GijMul, const float GikMul,
                  int NPart) {
                  
    int i = threadindex;
	if (i < NPart) {
	    float T_i = Ti[i];
	    float T_j = Tj[i];
	    float T_k = Tk[i];
	    
	    float g = (gamma != NULL) ? gamma[i] * gammaMul : gammaMul;
	    float Q_i = (Q != NULL) ? Q[i] * QMul : QMul;
	    float G_ij = (Gij != NULL) ? Gij[i] * GijMul : GijMul;
	    float G_ik = (Gik != NULL) ? Gik[i] * GikMul : GikMul;
	    
	    float C_i = g * T_i;
	    if (C_i == 0.0f) {
	        Qi[i] = 0.0;
	    }
	    
	    Qi[i] = (-G_ij * (T_i - T_j) - G_ik * (T_i - T_k) + Q_i) / (g * T_i);
	}
}


__global__  void QKern(float* __restrict__ Qi,
                  const float* __restrict__ Ti, const float* __restrict__ Tj, const float* __restrict__ Tk,
                  const float* __restrict__ Ci,
                  const float* __restrict__ Gij, const float* __restrict__ Gik,
                  const float CMul,
                  const float GijMul, const float GikMul,
                  int NPart) {
                    
    int i = threadindex;
	if (i < NPart) {
	    float T_i = Ti[i];
	    float T_j = Tj[i];
	    float T_k = Tk[i];
	    
	    float C_i = (Ci != NULL) ? Ci[i] * CMul : CMul;
	    float G_ij = (Gij != NULL) ? Gij[i] * GijMul : GijMul;
	    float G_ik = (Gik != NULL) ? Gik[i] * GikMul : GikMul;
	    
	    if (C_i == 0.0f) {
	        Qi[i] = 0.0;
	    }
	    
	    Qi[i] = (-G_ij * (T_i - T_j) - G_ik * (T_i - T_k)) / C_i;
	}
}


__export__ void Qe_async(float** Qe,
                         float** Te, float** Tl, float** Ts, 
                         float** Q, 
                         float** gamma_e, 
                         float** Gel, float** Ges,
                         float QMul, 
                         float gamma_eMul, 
                         float GelMul, float GesMul, 
                         CUstream* stream, int Npart){
    dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Qe[dev] != NULL);
		assert(Te[dev] != NULL);
		assert(Tl[dev] != NULL);
		assert(Ts[dev] != NULL);

		gpu_safe(cudaSetDevice(deviceId(dev)));
		QeKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Qe[dev],
		                                                                Te[dev], Tl[dev], Ts[dev],
		                                                                Q[dev],
		                                                                gamma_e[dev],
		                                                                Gel[dev], Ges[dev],
		                                                                QMul,
		                                                                gamma_eMul, 
		                                                                GelMul, GesMul, 
		                                                                Npart);
		                                                                
	}
}
__export__ void Qs_async(float** Qs,
                         float** Te, float** Tl, float** Ts, 
                         float** Cs, 
                         float** Gsl, float** Ges, 
                         float CsMul, 
                         float GslMul, float GesMul, CUstream* stream, int Npart) {
    dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Qs[dev] != NULL);
		assert(Te[dev] != NULL);
		assert(Tl[dev] != NULL);
		assert(Ts[dev] != NULL);

		gpu_safe(cudaSetDevice(deviceId(dev)));
		QKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>>  (Qs[dev],
		                                                                Ts[dev], Te[dev], Tl[dev],
		                                                                Cs[dev],
		                                                                Ges[dev], Gsl[dev],
		                                                                CsMul, 
		                                                                GesMul, GslMul, 
		                                                                Npart);
	}
}
__export__ void Ql_async(float** Ql,
                         float** Te, float** Tl, float** Ts, 
                         float** Cl, 
                         float** Gel, float** Gsl,
                         float ClMul,
                         float GelMul, float GslMul, 
                         CUstream* stream, int Npart){
    dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Ql[dev] != NULL);
		assert(Te[dev] != NULL);
		assert(Tl[dev] != NULL);
		assert(Ts[dev] != NULL);
        
		gpu_safe(cudaSetDevice(deviceId(dev)));
		QKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>>  (Ql[dev],
		                                                                Tl[dev], Te[dev], Ts[dev],
		                                                                Cl[dev],
		                                                                Gel[dev], Gsl[dev],
		                                                                ClMul, 
		                                                                GelMul, GslMul, 
		                                                                Npart);
	}
}

#ifdef __cplusplus
}
#endif
