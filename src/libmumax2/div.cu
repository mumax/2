#include "div.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void divKern(float* dst, float* a, float* b, int Npart) {
	int i = threadindex;
	if (i < Npart) {

        float bb = (b == NULL) ? 1.0f : b[i];
		dst[i] = (bb == 0.0f)? 0.0f : a[i] / bb;
	}
}


__export__ void divAsync(float** dst, float** a, float** b, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		divKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev], a[dev], b[dev], Npart);
	}
}

#ifdef __cplusplus
}
#endif

