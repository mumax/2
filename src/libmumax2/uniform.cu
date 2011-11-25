#include "uniform.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void initScalarQuantUniformRegionKern(float* S, float* regions, float* initValues, int regionNum, int Npart) {
	int i = threadindex;
	if (i < Npart) {
		int regionIndex = __float2int_rn(regions[i]);
		if (regionIndex < regionNum) {
			S[i] = initValues[regionIndex];
		}
		else {
			S[i] = -1.0f;
		}
	}
}


void initScalarQuantUniformRegionAsync(float** S, float** regions, float* host_initValues, int initValNum, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(S[dev] != NULL);
		assert(regions[dev] != NULL);
		assert(host_initValues != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		float* dev_initValues;
		cudaMalloc( (void**)&dev_initValues,initValNum * sizeof(float));
		cudaMemcpy(dev_initValues,host_initValues,initValNum * sizeof(float), cudaMemcpyHostToDevice);
		initScalarQuantUniformRegionKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (S[dev],regions[dev],dev_initValues, initValNum, Npart);
		cudaFree(dev_initValues);
	}
}

///@internal
__global__ void initVectorQuantUniformRegionKern(float* Sx, float* Sy, float* Sz,
												 float* regions,
												 float* initValuesX, float* initValuesY, float* initValuesZ,
												 int regionNum,
												 int Npart) {
	int i = threadindex;
	if (i < Npart) {
		int regionIndex = __float2int_rn(regions[i]);
		if (regionIndex < regionNum) {
			Sx[i] = initValuesX[regionIndex];
			Sy[i] = initValuesY[regionIndex];
			Sz[i] = initValuesZ[regionIndex];
		}
	}
}


void initVectorQuantUniformRegionAsync(float** Sx, float** Sy, float** Sz, float** regions, float* initValuesX, float* initValuesY, float* initValuesZ, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	int initValNum = sizeof(initValuesX) / sizeof(float);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Sx[dev] != NULL);
		assert(Sy[dev] != NULL);
		assert(Sz[dev] != NULL);
		assert(regions[dev] != NULL);
		assert(initValuesX != NULL);
		assert(initValuesY != NULL);
		assert(initValuesZ != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		initVectorQuantUniformRegionKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Sx[dev],Sy[dev],Sz[dev],regions[dev],initValuesX,initValuesY,initValuesZ, initValNum, Npart);
	}
}

#ifdef __cplusplus
}
#endif
