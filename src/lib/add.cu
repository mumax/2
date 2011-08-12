
#include "add.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void addKern(float *dst, float *a, float *b, int Npart) {
	int i = threadindex;
	if (i < Npart) {
		dst[i] = a[i] + b[i];
	}
}

void add(float **dst, float **a, float **b, CUstream * stream, int Ndev, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int i = 0; i < Ndev; i++) {
		addKern <<< gridSize, blockSize, 0, cudaStream_t(stream[i]) >>> (dst[i], a[i], b[i], Npart);
	}
	for (int i = 0; i < Ndev; i++) {
		cuStreamSynchronize(stream[i]);
	}
}

#ifdef __cplusplus
}
#endif
