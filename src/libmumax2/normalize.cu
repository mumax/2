#include "normalize.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void normalizeKern(float* mx, float* my, float* mz, 
						   float* norm_map, int Npart) {
	int i = threadindex;
	if (i < Npart) {

		// reconstruct norm from map
		float norm;
		if(norm_map == NULL){
			norm = 1.0f;
		}else{
			norm = norm_map[i];
		}

    	float Mx = mx[i];
    	float My = my[i];
    	float Mz = mz[i];
    
		float Mnorm = sqrtf(Mx*Mx + My*My + Mz*Mz);
		float scale;
		if (Mnorm != 0.f){
			scale = norm / Mnorm;
			scale = 1.0f / Mnorm;
		}else{
			scale = 0.f;
		}

		mx[i] = Mx * scale;
		my[i] = My * scale;
		mz[i] = Mz * scale;
	}
}


__export__ void normalizeAsync(float** mx, float** my, float** mz, float** norm_map, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(mx[dev] != NULL);
		assert(my[dev] != NULL);
		assert(mz[dev] != NULL);
		// normMap may be null
		gpu_safe(cudaSetDevice(deviceId(dev)));
		normalizeKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (mx[dev],my[dev],mz[dev], norm_map[dev], Npart);
	}
}

#ifdef __cplusplus
}
#endif
