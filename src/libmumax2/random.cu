#include "random.h"

#include "multigpu.h"
#include <cuda.h>
#include <curand_kernel.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SEED 123
#define OFFSET 0


///@internal
__global__ void setUpRandomRegionKern(curandState* state)
{
    int i = threadindex;
    curand_init(SEED, i, OFFSET, &state[i]);
}

///@internal
__global__ void initScalarQuantRandomUniformRegionKern(float* S,
        float* regions,
        bool* regionsToProceed,
        int regionNum,
        int Npart,
        curandState* globalState,
        float max, float min)
{
    int i = threadindex;
    if (i < Npart)
    {
        int regionIndex = __float2int_rn(regions[i]);
        if (regionIndex < regionNum && regionIndex > 0 && regionsToProceed[regionIndex] == true)
        {
            curandState localState = globalState[i];
            S[i] = min + (max - min) * curand_uniform(&localState);
            globalState[i] = localState;
        }
    }
}


__export__ void initScalarQuantRandomUniformRegionAsync(float** S,
        float** regions,
        bool* host_regionsToProceed,
        int regionNum,
        CUstream* stream,
        int Npart,
        float max, float min)
{
    assert(max != min);
    if (max < min)
    {
        float tmp = min;
        min = max;
        max = tmp;
    }
    curandState* devState;
    bool* dev_regionsToProceed;
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    for (int dev = 0; dev < nDevice(); dev++)
    {
        assert(S[dev] != NULL);
        assert(regions[dev] != NULL);
        assert(host_regionsToProceed != NULL);
        gpu_safe(cudaSetDevice(deviceId(dev)));
        gpu_safe( cudaMalloc( (void**)&dev_regionsToProceed, regionNum * sizeof(bool)));
        gpu_safe( cudaMemcpy(dev_regionsToProceed, host_regionsToProceed, regionNum * sizeof(bool), cudaMemcpyHostToDevice));
        gpu_safe( cudaMalloc( (void**)&devState, Npart * sizeof(curandState)));
        setUpRandomRegionKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (devState);
        initScalarQuantRandomUniformRegionKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (S[dev], regions[dev], dev_regionsToProceed, regionNum, Npart, devState, max, min);
        cudaFree(dev_regionsToProceed);
        cudaFree(devState);
    }
}

///@internal
__global__ void initVectorQuantRandomUniformRegionKern(float* Sx, float* Sy, float* Sz,
        float* regions,
        bool* regionsToProceed,
        int regionNum,
        int Npart,
        curandState* globalState)
{
    int i = threadindex;
    if (i < Npart)
    {
        int regionIndex = __float2int_rn(regions[i]);
        if (regionIndex < regionNum && regionIndex > 0 && regionsToProceed[regionIndex] == true)
        {
            curandState localState = globalState[i];
            Sx[i] = 2.0f * curand_uniform(&localState) - 1.0f;
            Sy[i] = 2.0f * curand_uniform(&localState) - 1.0f;
            Sz[i] = 2.0f * curand_uniform(&localState) - 1.0f;
            float norm = sqrt(Sx[i] * Sx[i] + Sy[i] * Sy[i] + Sz[i] * Sz[i]);
            Sx[i] /= norm;
            Sy[i] /= norm;
            Sz[i] /= norm;
            globalState[i] = localState;
        }
    }
}


__export__ void initVectorQuantRandomUniformRegionAsync(float** Sx, float** Sy, float** Sz,
        float** regions,
        bool* host_regionsToProceed,
        int regionNum,
        CUstream* stream,
        int Npart)
{
    curandState* devState;
    bool* dev_regionsToProceed;
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    for (int dev = 0; dev < nDevice(); dev++)
    {
        assert(Sx[dev] != NULL);
        assert(Sy[dev] != NULL);
        assert(Sz[dev] != NULL);
        assert(regions[dev] != NULL);
        assert(host_regionsToProceed != NULL);
        gpu_safe(cudaSetDevice(deviceId(dev)));
        gpu_safe( cudaMalloc( (void**)&dev_regionsToProceed, regionNum * sizeof(bool)));
        gpu_safe( cudaMemcpy(dev_regionsToProceed, host_regionsToProceed, regionNum * sizeof(bool), cudaMemcpyHostToDevice));
        gpu_safe( cudaMalloc( (void**)&devState, Npart * sizeof(curandState)));
        setUpRandomRegionKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (devState);
        initVectorQuantRandomUniformRegionKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Sx[dev], Sy[dev], Sz[dev], regions[dev], dev_regionsToProceed, regionNum, Npart, devState);
        cudaFree(dev_regionsToProceed);
        cudaFree(devState);
    }
}

#ifdef __cplusplus
}
#endif
