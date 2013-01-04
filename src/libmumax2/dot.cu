#include "gilbert.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void dotKern(float* dst,
                        float* ax, float* ay, float* az, 
                        float* bx, float* by, float* bz, 
						int Npart) {
	int i = threadindex;
	if (i < Npart) {
	    float3 a = make_float3(ax[i], ay[i], az[i]);
	    float3 b = make_float3(bx[i], by[i], bz[i]);
        dst[i] = dotf(a, b); 
	}
}

///@internal
__global__ void dotSignKern(float* dst,
                        float* ax, float* ay, float* az, 
                        float* bx, float* by, float* bz, 
						int Npart) {
	int i = threadindex;
	if (i < Npart) {
	    float3 a = make_float3(ax[i], ay[i], az[i]);
	    float3 b = make_float3(bx[i], by[i], bz[i]);
	    // get sign of b
	    float3 udir = make_float3(1.0f, 1.0f, 1.0f); 
	    float sign = dotf(b, udir) / len(b);
        dst[i] = sign * dotf(a, b); 
	}
}


__export__ void dotAsync(float** dst, float** ax, float** ay, float** az, float** bx, float** by, float** bz, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(dst[dev] != NULL);
		assert(ax[dev] != NULL);
		assert(ay[dev] != NULL);
		assert(az[dev] != NULL);
		assert(bx[dev] != NULL);
		assert(by[dev] != NULL);
		assert(bz[dev] != NULL);
		// alphaMap may be null
		gpu_safe(cudaSetDevice(deviceId(dev)));
		dotKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev], ax[dev],ay[dev],az[dev], bx[dev],by[dev],bz[dev], Npart);
	}
}

__export__ void dotSignAsync(float** dst, float** ax, float** ay, float** az, float** bx, float** by, float** bz, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(dst[dev] != NULL);
		assert(ax[dev] != NULL);
		assert(ay[dev] != NULL);
		assert(az[dev] != NULL);
		assert(bx[dev] != NULL);
		assert(by[dev] != NULL);
		assert(bz[dev] != NULL);
		// alphaMap may be null
		gpu_safe(cudaSetDevice(deviceId(dev)));
		dotSignKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev], ax[dev],ay[dev],az[dev], bx[dev],by[dev],bz[dev], Npart);
	}
}


#ifdef __cplusplus
}
#endif
