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
__global__ void limiterKern(float* __restrict__ Mx,float* __restrict__ My, float* __restrict__ Mz, 
						      float limit, 
						      int Npart) {
	int i = threadindex;
	if (i < Npart) {

		// reconstruct norm from map
		real3 M = make_real3(Mx[i], My[i], Mz[i]);
		
		real Ms = len(M);
		
		if (Ms == 0.0) {
		    Mx[i] = 0.0f;
		    My[i] = 0.0f;
		    Mz[i] = 0.0f;
		    return;
		}
				
		real norm = (Ms > 1.0) ? 1.0 / Ms : 1.0;
		
		Mx[i] = M.x * norm;
		My[i] = M.y * norm;
		Mz[i] = M.z * norm;
		     	
	}
}


__export__ void limiterAsync(float** Mx, float** My, float** Mz,
                               float limit,
                               CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		limiterKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Mx[dev],My[dev],Mz[dev],
		                                                                       limit,
		                                                                       Npart);
	}
}

#ifdef __cplusplus
}
#endif
