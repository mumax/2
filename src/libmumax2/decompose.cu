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
__global__ void decomposeKern(float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
                              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                              float* __restrict__ msat,
                              float msatMul,
                              int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        // reconstruct norm from map

        float3 M = make_float3(Mx[i], My[i], Mz[i]);

        float Ms = len(M);

        if (Ms == 0.0f)
        {
            mx[i] = 0.0f;
            my[i] = 0.0f;
            mz[i] = 0.0f;
            msat[i] = 0.0f;
            return;
        }

        mx[i] = M.x / Ms;
        my[i] = M.y / Ms;
        mz[i] = M.z / Ms;

        msat[i] = Ms;
    }
}


__export__ void decomposeAsync(float** Mx, float** My, float** Mz,
                               float** mx, float** my, float** mz,
                               float** msat,
                               float msatMul,
                               CUstream* stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    for (int dev = 0; dev < nDevice(); dev++)
    {
        assert(mx[dev] != NULL);
        assert(my[dev] != NULL);
        assert(mz[dev] != NULL);
        // normMap may be null
        gpu_safe(cudaSetDevice(deviceId(dev)));
        decomposeKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Mx[dev], My[dev], Mz[dev],
                mx[dev], my[dev], mz[dev],
                msat[dev],
                msatMul,
                Npart);
    }
}

#ifdef __cplusplus
}
#endif
