#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

///@internal
    __global__ void dotKern(float* __restrict__ dst,
                            float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                            float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
                            int Npart)
    {
        int i = threadindex;
        if (i < Npart)
        {
            float3 a = make_float3(ax[i], ay[i], az[i]);
            float3 b = make_float3(bx[i], by[i], bz[i]);
            dst[i] = dotf(a, b);
        }
    }



///@internal
// if b || c, then dst < 0.
    __global__ void dotSignKern(float* __restrict__ dst,
                                float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                                float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
                                float* __restrict__ cx, float* __restrict__ cy, float* __restrict__ cz,
                                int Npart)
    {
        int i = threadindex;
        if (i < Npart)
        {
            float3 a = make_float3(ax[i], ay[i], az[i]);
            float3 b = make_float3(bx[i], by[i], bz[i]);
            float3 c = make_float3(cx[i], cy[i], cz[i]);
            float dotP = dotf(a, b);
            float sign = -dotf(b, c); // !!!
            dst[i] = copysign(dotP, sign);
        }
    }


    __export__ void dotAsync(float** dst, float** ax, float** ay, float** az, float** bx, float** by, float** bz, CUstream* stream, int Npart)
    {
        dim3 gridSize, blockSize;
        make1dconf(Npart, &gridSize, &blockSize);
        for (int dev = 0; dev < nDevice(); dev++)
        {
            assert(dst[dev] != NULL);
            assert(ax[dev] != NULL);
            assert(ay[dev] != NULL);
            assert(az[dev] != NULL);
            assert(bx[dev] != NULL);
            assert(by[dev] != NULL);
            assert(bz[dev] != NULL);
            // alphaMap may be null
            gpu_safe(cudaSetDevice(deviceId(dev)));
            dotKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev], ax[dev], ay[dev], az[dev], bx[dev], by[dev], bz[dev], Npart);
        }
    }

    __export__ void dotSignAsync(float** dst, float** ax, float** ay, float** az, float** bx, float** by, float** bz, float** cx, float** cy, float** cz, CUstream* stream, int Npart)
    {
        dim3 gridSize, blockSize;
        make1dconf(Npart, &gridSize, &blockSize);
        for (int dev = 0; dev < nDevice(); dev++)
        {
            assert(dst[dev] != NULL);
            assert(ax[dev] != NULL);
            assert(ay[dev] != NULL);
            assert(az[dev] != NULL);
            assert(bx[dev] != NULL);
            assert(by[dev] != NULL);
            assert(bz[dev] != NULL);
            assert(cx[dev] != NULL);
            assert(cy[dev] != NULL);
            assert(cz[dev] != NULL);
            // alphaMap may be null
            gpu_safe(cudaSetDevice(deviceId(dev)));
            dotSignKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev],
                    ax[dev], ay[dev], az[dev],
                    bx[dev], by[dev], bz[dev],
                    cx[dev], cy[dev], cz[dev],
                    Npart);
        }
    }


#ifdef __cplusplus
}
#endif
