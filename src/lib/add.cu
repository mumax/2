
#include "add.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void addKern(float* dst, float* a, float* b, int Npart) {
	int i = threadindex;
	if (i < Npart) {
		dst[i] = a[i] + b[i];
	}
}


void addAsync(float** dst, float** a, float** b, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int i = 0; i < nDevice(); i++) {
		gpu_safe(cudaSetDevice(deviceId(i)));
		addKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (dst[i], a[i], b[i], Npart);
	}
}



///@internal
__global__ void maddKern(float* dst, float* a, float* b, float mulB, int Npart) {
	int i = threadindex;
	if (i < Npart) {
		dst[i] = a[i] + mulB * b[i];
	}
}


void maddAsync(float** dst, float** a, float** b, float mulB, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int i = 0; i < nDevice(); i++) {
		gpu_safe(cudaSetDevice(deviceId(i)));
		maddKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (dst[i], a[i], b[i], mulB, Npart);
	}
}

#ifdef __cplusplus
}
#endif
