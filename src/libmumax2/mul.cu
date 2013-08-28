#include "mul.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void mulKern(float* dst, float* a, float* b, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] * b[i];
    }
}


__export__ void mulAsync(float** dst, float** a, float** b, CUstream* stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    for (int dev = 0; dev < nDevice(); dev++)
    {
        gpu_safe(cudaSetDevice(deviceId(dev)));
        mulKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (dst[dev], a[dev], b[dev], Npart);
    }
}

#ifdef __cplusplus
}
#endif

