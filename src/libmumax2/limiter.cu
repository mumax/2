#include "normalize.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void limiterKern(float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz, 
                            float* __restrict__ limitMask,
                            float msatMul,
			    float limitMul, 
			    int Npart) {
	int i = threadindex;
	
	if (i < Npart) {

		real3 M = make_real3(Mx[i], My[i], Mz[i]);

		real nMn = len(M);

		real limit = (limitMask != NULL) ? limitMask[i] * limitMul : limitMul;

		if (nMn == 0.0 || limit == 0.0) {
		    Mx[i] = 0.0f;
		    My[i] = 0.0f;
		    Mz[i] = 0.0f;
		    return;
		}
	    		
	    real Ms = msatMul * nMn;
	       			
		real ratio = Ms / limit;

		real pre = 1.0 / (ratio * nMn);        

		real norm = (ratio > 1.0) ? pre : 1.0;
		
		Mx[i] = M.x * norm;
		My[i] = M.y * norm;
		Mz[i] = M.z * norm;	
	}
}


__export__ void limiterAsync(float** Mx, float** My, float** Mz,
                            float** limitMask,
                            float msatMul,
                            float limitMul,
                            CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		limiterKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Mx[dev],My[dev],Mz[dev],
		                                                                     limitMask[dev],
		                                                                     msatMul,
		                                                                     limitMul,
		                                                                     Npart);
	}
}

#ifdef __cplusplus
}
#endif
