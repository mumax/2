#include "kappa.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal


__global__ void kappaKern(float* __restrict__ kappa,
                          float* __restrict__ msat0Msk,
                          float* __restrict__ msat0T0Msk,
                          float* __restrict__ T,
                          float* __restrict__ TcMsk,
                          float* __restrict__ SMsk,
                          float* __restrict__ nMsk,
                          const float msat0Mul,
                          const float msat0T0Mul,
                          const float TcMul,
                          const float SMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        float msat0T0u = (msat0T0Msk == NULL) ? 1.0f : msat0T0Msk[i];
        float msat0T0 = msat0T0Mul * msat0T0u;

        float Temp = T[i];

        if (msat0T0 == 0.0f || Temp == 0.0f)
        {
            kappa[i] = 0.0f;
            return;
        }


        float S = (SMsk == NULL) ? SMul : SMul * SMsk[i];
        float Tc = (TcMsk == NULL) ? TcMul : TcMul * TcMsk[i];
        float msat0 = (msat0Msk == NULL) ? msat0Mul : msat0Mul * msat0Msk[i];
        float J0  = 3.0f * Tc / (S * (S + 1.0f)); // in h^2 units
        float n = (nMsk == NULL) ? 1.0f : nMsk[i];

        float mul = msat0T0u * msat0T0u / (S * S * J0 * n); // msat0T0 mul should be in the kappa multiplier
        float me = msat0 / msat0T0;
        float b = S * S * J0 / Temp;
        float meb = me * b;
        float f = b * dBjdxf(S, meb);
        float k = mul * (f / (1.0f - f));
        kappa[i] = k;
    }
}

__export__ void kappaAsync(float** kappa,
                           float** msat0,
                           float** msat0T0,
                           float** T,
                           float** Tc,
                           float** S,
                           float** n,
                           const float msat0Mul,
                           const float msat0T0Mul,
                           const float TcMul,
                           const float SMul,
                           int Npart,
                           CUstream* stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    for (int dev = 0; dev < nDevice(); dev++)
    {
        assert(kappa[dev] != NULL);
        assert(msat0[dev] != NULL);
        assert(T[dev] != NULL);
        gpu_safe(cudaSetDevice(deviceId(dev)));
        kappaKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (kappa[dev],
                msat0[dev],
                msat0T0[dev],
                T[dev],
                Tc[dev],
                S[dev],
                n[dev],
                msat0Mul,
                msat0T0Mul,
                TcMul,
                SMul,
                Npart);
    }

}

#ifdef __cplusplus
}
#endif
