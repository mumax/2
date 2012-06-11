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
__global__ void normalizeKern(float* mx, float* my, float* mz, 
						   float* norm_map, int Npart) {
	int i = threadindex;
	if (i < Npart) {

		// reconstruct norm from map
		real norm;
		if(norm_map == NULL){
			norm = 1.0f;
		}else{
			norm = norm_map[i];
		}

    	real Mx = mx[i];
    	real My = my[i];
    	real Mz = mz[i];
    
		real Mnorm = sqrt(Mx*Mx + My*My + Mz*Mz);
		real scale;
		if (Mnorm != 0.0){
			//scale = norm / Mnorm;
			scale = 1.0 / Mnorm;
		}else{
			scale = 0.0;
		}
        real m_x = Mx * scale;
        real m_y = My * scale;
        real m_z = Mz * scale;
        
		mx[i] = m_x;
		my[i] = m_y;
		mz[i] = m_z;
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
