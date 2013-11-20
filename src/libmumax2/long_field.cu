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
                                float* __restrict__ msat0T0Msk,
                                float* __restrict__ SMsk,
                                float* __restrict__ kappaMsk,
                                float* __restrict__ TcMsk,
                                float* __restrict__ TsMsk,
                                float msat0T0Mul,
                                float SMul,
                                float kappaMul,
                                float TcMul,
                                float TsMul,
                                int NPart)
{

    int I = threadindex;

    if (I < NPart)  // Thread configurations are usually too large...
    {

        float Ms0T0 = msat0T0Mul * getMaskUnity(msat0T0Msk, I);
        float S = SMul * getMaskUnity(SMsk, I);
        float kappa = kappaMul * getMaskUnity(kappaMsk, I);
        float Tc = TcMul * getMaskUnity(TcMsk, I);
        float Ts = TsMul * getMaskUnity(TsMsk, I);

        if (Ms0T0 == 0.0f || Ts == Tc || kappa == 0.0f)
        {
            hx[I] = 0.0f;
            hy[I] = 0.0f;
            hz[I] = 0.0f;
            return;
        }


        float3 mf = make_float3(mx[I], my[I], mz[I]);
        float3 s = normalize(mf);

        float abs_mf = len(mf);

        float J0  = 3.0f * Tc / (S * (S + 1.0f));

        float b = S * S * J0 / Ts;

        float meb = abs_mf * b;

        float M = Ms0T0 * abs_mf;

        float M0 = Ms0T0 * Bj(S, meb);

        float mult = (M0 - M) / (b * kappa * dBjdxf(S, meb));

        hx[I] = mult * s.x;
        hy[I] = mult * s.y;
        hz[I] = mult * s.z;

    }
}


__export__ void long_field_async(float** hx, float** hy, float** hz,
                                 float** mx, float** my, float** mz,
                                 float** msat0T0,
                                 float** S,
                                 float** kappa,
                                 float** Tc,
                                 float** Ts,
                                 float msat0T0Mul,
                                 float SMul,
                                 float kappaMul,
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
            msat0T0[dev],
            S[dev],
            kappa[dev],
            Tc[dev],
            Ts[dev],
            msat0T0Mul,
            SMul,
            kappaMul,
            TcMul,
            TsMul,
            NPart);
}

// ========================================

#ifdef __cplusplus
}
#endif
