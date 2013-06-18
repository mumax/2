#include "baryakhtar-transverse.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

    __global__ void baryakhtarTransverseKernFloat(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
            float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
            float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

            float* __restrict__ msat0T0Msk,

            float* __restrict__ mu_xx,
            float* __restrict__ mu_yy,
            float* __restrict__ mu_zz,

            const float muMul_xx,
            const float muMul_yy,
            const float muMul_zz,

            int Npart)
    {

        int x0 = threadindex;

        if (x0 < Npart)
        {

            float msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];

            // make sure there is no torque in vacuum!
            if (msat0T0 == 0.0f)
            {
                tx[x0] = 0.0f;
                ty[x0] = 0.0f;
                tz[x0] = 0.0f;
                return;
            }

            float3 H = make_float3(hx[x0], hy[x0], hz[x0]);
            float3 M = make_float3(Mx[x0], My[x0], Mz[x0]);

            float3 MxH = crossf(M, H);

            float3 mu_MxH;

            float m_xx = (mu_xx != NULL) ? mu_xx[x0] * muMul_xx : muMul_xx;

            mu_MxH.x = m_xx * MxH.x;

            float m_yy = (mu_yy != NULL) ? mu_yy[x0] * muMul_yy : muMul_yy;

            mu_MxH.y = m_yy * MxH.y;

            float m_zz = (mu_zz != NULL) ? mu_zz[x0] * muMul_zz : muMul_zz;

            mu_MxH.z = m_zz * MxH.z;

            float3 _Mxmu_MxH = crossf(mu_MxH, M);

            tx[x0] = _Mxmu_MxH.x;
            ty[x0] = _Mxmu_MxH.y;
            tz[x0] = _Mxmu_MxH.z;
        }
    }

    __export__  void baryakhtar_transverse_async(float** tx, float**  ty, float**  tz,
            float**  Mx, float**  My, float**  Mz,
            float**  hx, float**  hy, float**  hz,

            float** msat0T0,

            float** mu_xx,
            float** mu_yy,
            float** mu_zz,

            const float muMul_xx,
            const float muMul_yy,
            const float muMul_zz,

            CUstream* stream,
            int Npart)
    {
        dim3 gridSize, blockSize;
        make1dconf(Npart, &gridSize, &blockSize);
        int nDev = nDevice();

        for (int dev = 0; dev < nDev; dev++)
        {

            gpu_safe(cudaSetDevice(deviceId(dev)));
            // calculate dev neighbours

            baryakhtarTransverseKernFloat <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],
                    Mx[dev], My[dev], Mz[dev],
                    hx[dev], hy[dev], hz[dev],

                    msat0T0[dev],

                    mu_xx[dev],
                    mu_yy[dev],
                    mu_zz[dev],

                    muMul_xx,
                    muMul_yy,
                    muMul_zz,

                    Npart);
        } // end dev < nDev loop

    }

    // ========================================

#ifdef __cplusplus
}
#endif
