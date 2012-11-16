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
		float norm = (norm_map != NULL ) ? norm_map[i] : 1.0f;
		
		float Mx = mx[i];
		float My = my[i];
		float Mz = mz[i];
    
		float Mnorm = sqrtf(Mx * Mx + My * My + Mz * Mz);
		float scale = (norm == 0.0f) ? 0.0f : 1.0f;
		
		scale = (Mnorm != 0.0f) ? scale/ Mnorm : 0.0;
		
		float m_x = Mx * scale;
		float m_y = My * scale;
		float m_z = Mz * scale;
        
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
