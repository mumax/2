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

		// reconstruct inverse norm from map
		float invnorm;
		if(norm_map == NULL){
			invnorm = 1.0f;
		}else{
			invnorm = norm_map[i];
			if(invnorm != 0.0f){
				invnorm = 1.0f/invnorm;
			}
		}

    	float Mx = mx[i];
    	float My = my[i];
    	float Mz = mz[i];
    
		invnorm = invnorm * (1.0f/sqrtf(Mx*Mx + My*My + Mz*Mz));	
		mx[i] = Mx * invnorm;
		my[i] = My * invnorm;
		mz[i] = Mz * invnorm;
	}
}


void normalizeAsync(float** mx, float** my, float** mz, float** norm_map, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int i = 0; i < nDevice(); i++) {
		assert(mx[i] != NULL);
		assert(my[i] != NULL);
		assert(mz[i] != NULL);
		// normMap may be null
		gpu_safe(cudaSetDevice(deviceId(i)));
		normalizeKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (mx[i],my[i],mz[i], norm_map[i], Npart);
	}
}

#ifdef __cplusplus
}
#endif
