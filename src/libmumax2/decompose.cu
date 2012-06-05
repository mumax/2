#include "normalize.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

extern "C" {

///@internal
__global__ void decomposeKern(float* Mx, float* My, float* Mz,
                              float* mx, float* my, float* mz,
                              float* msat,   
						      float msatMul, 
						      int Npart) {
	int i = threadindex;
	if (i < Npart) {

		// reconstruct norm from map
		real3 M = make_real3(Mx[i], My[i], Mz[i]);
		real Ms = len(M);
		
		if (Ms == 0.0) {
		    mx[i] = 0.0;
		    my[i] = 0.0;
		    mz[i] = 0.0;
		    msat[i] = 0.0;
		    return; 
		}

    	mx[i] = M.x / Ms;
    	my[i] = M.y / Ms;
    	mz[i] = M.z / Ms;
        
        msat[i] = Ms;// / msatMul;
        
	}
}


__export__ void decomposeAsync(float** Mx, float** My, float** Mz,
                               float** mx, float** my, float** mz, 
                               float** msat, 
                               float msatMul,
                               CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(mx[dev] != NULL);
		assert(my[dev] != NULL);
		assert(mz[dev] != NULL);
		// normMap may be null
		gpu_safe(cudaSetDevice(deviceId(dev)));
		decomposeKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Mx[dev],My[dev],Mz[dev],
		                                                                       mx[dev],my[dev],mz[dev], 
		                                                                       msat[dev], 
		                                                                       msatMul,
		                                                                       Npart);
	}
}

}
