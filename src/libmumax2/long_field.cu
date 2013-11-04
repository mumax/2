#include "long_field.h"
#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"


#ifdef __cplusplus
extern "C" {
#endif
// ========================================

__global__ void long_field_Kern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
                                float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                                float* __restrict__ msat0Msk,
                                float* __restrict__ msat0T0Msk,
                                float* __restrict__ kappaMsk,
                                float* __restrict__ TcMsk,
                                float* __restrict__ TsMsk,
                                float kappaMul,
                                float msat0Mul,
                                float msat0T0Mul,
                                float TcMul,
                                float TsMul,
                                int NPart)
{

    int I = threadindex;

    if (I < NPart)  // Thread configurations are usually too large...
    {

        float Ms0T0 = (msat0T0Msk != NULL ) ? msat0T0Msk[I] * msat0T0Mul : msat0T0Mul;
        float Ms0 = (msat0Msk != NULL ) ? msat0Msk[I] * msat0Mul : msat0Mul;
        float kappa = (kappaMsk != NULL ) ? kappaMsk[I] * kappaMul : kappaMul;
        float Tc = (TcMsk != NULL) ? TcMsk[I] * TcMul : TcMul;
        float Ts = (TsMsk != NULL) ? TsMsk[I] * TsMul : TsMul;

        if (Ms0T0 == 0.0f || kappa == 0.0f || Ts == Tc)
        {
            hx[I] = 0.0f;
            hy[I] = 0.0f;
            hz[I] = 0.0f;
            return;
        }

        kappa = 1.0f / kappa;

        float3 mf = make_float3(mx[I], my[I], mz[I]);

        float mf2 = dotf(mf, mf);

        float Mf2 = Ms0T0 * Ms0T0 * mf2;

        float Ms02 = Ms0 * Ms0;

        float dM2 = (Ms02 - Mf2);

        float mult = (Ts < Tc) ? dM2 / Ms02 : - 2.0f * (1.0f + 0.6f * mf2 * Tc / (Ts - Tc)); // 2.0 is to account kappa = 0.5 / kappa

        mult = (mult == 0.0f) ? 0.0f : kappa * Ms0T0 * mult;

        hx[I] = mult * mf.x;
        hy[I] = mult * mf.y;
        hz[I] = mult * mf.z;

    }
}


__export__ void long_field_async(float** hx, float** hy, float** hz,
                                 float** mx, float** my, float** mz,
                                 float** msat0,
                                 float** msat0T0,
                                 float** kappa,
                                 float** Tc,
                                 float** Ts,
                                 float kappaMul,
                                 float msat0Mul,
                                 float msat0T0Mul,
                                 float TcMul,
                                 float TsMul,
                                 int NPart,
                                 CUstream* stream)
{
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    int dev = 0;
    gpu_safe(cudaSetDevice(deviceId(dev)));
    long_field_Kern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (hx[dev], hy[dev], hz[dev],
            mx[dev], my[dev], mz[dev],
            msat0[dev],
            msat0T0[dev],
            kappa[dev],
            Tc[dev],
            Ts[dev],
            kappaMul,
            msat0Mul,
            msat0T0Mul,
            TcMul,
            TsMul,
            NPart);
}

// ========================================

#ifdef __cplusplus
}
#endif
