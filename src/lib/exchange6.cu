#include "exchange6.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void exch6Kern(float* dst, float* a, float* b, int Npart) {
	int i = threadindex;
	if (i < Npart) {

	}
}


void exchange6Async(float** dst, float** a, float** b, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int i = 0; i < nDevice(); i++) {
		gpu_safe(cudaSetDevice(deviceId(i)));
		exch6Kern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (dst[i], a[i], b[i], Npart);
	}
}



#ifdef __cplusplus
}
#endif
