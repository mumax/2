#include "long_field.h"
#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
//#include "common_func.h"


#ifdef __cplusplus
extern "C" {
#endif
// ========================================

__global__ void long_field_Kern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
                                float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                                float* __restrict__ msatMsk,
                                float* __restrict__ msat0Msk,
                                float* __restrict__ msat0T0Msk,
                                float* __restrict__ kappaMsk,
                                float* __restrict__ TcMsk,
                                float* __restrict__ TsMsk,
                                float kappaMul,
                                float msatMul,
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
        // ~ if (I == 0) {
        // ~ printf("Ms0T0: %f\tMs0: %f\tkappa: %f\n", Ms0T0, Ms0, kappa);
        // ~ }
        if (Ms0T0 == 0.0f || kappa == 0.0f || Ts == Tc)
        {
            hx[I] = 0.0f;
            hy[I] = 0.0f;
            hz[I] = 0.0f;
            return;
        }

        kappa = 1.0f / kappa;

        float Ms = (msatMsk != NULL ) ? msatMsk[I] * msatMul : msatMul;

        float3 m = make_float3(mx[I], my[I], mz[I]);

        float ratio = (Ts < Tc) ? Ms / Ms0 : Ms / Ms0T0;

        float mult = (Ts < Tc) ?          (1.0f - ratio * ratio)
                     : - 2.0f * (1.0f + 0.6f * ratio * ratio * Tc / (Ts - Tc)); // 2.0 is to account kappa = 0.5 / kappa

        mult = (mult == 0.0f) ? 0.0f : kappa * Ms * mult;

        hx[I] = mult * m.x;
        hy[I] = mult * m.y;
        hz[I] = mult * m.z;

    }
}


__export__ void long_field_async(float** hx, float** hy, float** hz,
                                 float** mx, float** my, float** mz,
                                 float** msat,
                                 float** msat0,
                                 float** msat0T0,
                                 float** kappa,
                                 float** Tc,
                                 float** Ts,
                                 float kappaMul,
                                 float msatMul,
                                 float msat0Mul,
                                 float msat0T0Mul,
                                 float TcMul,
                                 float TsMul,
                                 int NPart,
                                 CUstream* stream)
{
    //printf("NPart is: %d\n", NPart);
    // 1D configuration
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    int nDev = nDevice();
    for (int dev = 0; dev < nDev; dev++)
    {
        gpu_safe(cudaSetDevice(deviceId(dev)));
        long_field_Kern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (hx[dev], hy[dev], hz[dev],
                mx[dev], my[dev], mz[dev],
                msat[dev],
                msat0[dev],
                msat0T0[dev],
                kappa[dev],
                Tc[dev],
                Ts[dev],
                kappaMul,
                msatMul,
                msat0Mul,
                msat0T0Mul,
                TcMul,
                TsMul,
                NPart);
    } // end dev < nDev loop


}

// ========================================

#ifdef __cplusplus
}
#endif
