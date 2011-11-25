#include "uniform.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <stdio.h>

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
	}
}


void initScalarQuantUniformRegionAsync(float** S, float** regions, float* host_initValues, int initValNum, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	float* dev_initValues;
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(S[dev] != NULL);
		assert(regions[dev] != NULL);
		assert(host_initValues != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		gpu_safe( cudaMalloc( (void**)&dev_initValues,initValNum * sizeof(float)));
		gpu_safe( cudaMemcpy(dev_initValues,host_initValues,initValNum * sizeof(float), cudaMemcpyHostToDevice));
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
		if (regionIndex < regionNum && regionIndex > 0) {
			Sx[i] = initValuesX[regionIndex];
			Sy[i] = initValuesY[regionIndex];
			Sz[i] = initValuesZ[regionIndex];
		} else {
			Sx[i] = 1.0f;
			Sy[i] = 0.0f;
			Sz[i] = 0.0f;
		}
	}
}


void initVectorQuantUniformRegionAsync(float** Sx, float** Sy, float** Sz, float** regions, float* host_initValuesX, float* host_initValuesY, float* host_initValuesZ, int initValNum, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	float* dev_initValuesX;
	float* dev_initValuesY;
	float* dev_initValuesZ;
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(Sx[dev] != NULL);
		assert(Sy[dev] != NULL);
		assert(Sz[dev] != NULL);
		assert(regions[dev] != NULL);
		assert(host_initValuesX != NULL);
		assert(host_initValuesY != NULL);
		assert(host_initValuesZ != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		gpu_safe(cudaSetDevice(deviceId(dev)));
		gpu_safe( cudaMalloc( (void**)&dev_initValuesX,initValNum * sizeof(float)));
		gpu_safe( cudaMalloc( (void**)&dev_initValuesY,initValNum * sizeof(float)));
		gpu_safe( cudaMalloc( (void**)&dev_initValuesZ,initValNum * sizeof(float)));
		gpu_safe( cudaMemcpy(dev_initValuesX,host_initValuesX,initValNum * sizeof(float), cudaMemcpyHostToDevice));
		gpu_safe( cudaMemcpy(dev_initValuesY,host_initValuesY,initValNum * sizeof(float), cudaMemcpyHostToDevice));
		gpu_safe( cudaMemcpy(dev_initValuesZ,host_initValuesZ,initValNum * sizeof(float), cudaMemcpyHostToDevice));
		initVectorQuantUniformRegionKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Sx[dev],Sy[dev],Sz[dev],regions[dev],dev_initValuesX,dev_initValuesY,dev_initValuesZ, initValNum, Npart);
		cudaFree(dev_initValuesX);
		cudaFree(dev_initValuesY);
		cudaFree(dev_initValuesZ);
	}
}

#ifdef __cplusplus
}
#endif
