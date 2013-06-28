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
            float* __restrict__ Sx, float* __restrict__ Sy, float* __restrict__ Sz,
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
            float3 S = make_float3(Sx[x0], Sy[x0], Sz[x0]);

            // make sure there is no torque for non-magnetic points
            if (msat0T0 == 0.0f)
            {
                tx[x0] = 0.0f;
                ty[x0] = 0.0f;
                tz[x0] = 0.0f;
                return;
            }

            float3 H = make_float3(hx[x0], hy[x0], hz[x0]);

            float3 SxH = crossf(S, H);

            float3 mu_SxH;

            float m_xx = (mu_xx != NULL) ? mu_xx[x0] * muMul_xx : muMul_xx;

            mu_SxH.x = m_xx * SxH.x;

            float m_yy = (mu_yy != NULL) ? mu_yy[x0] * muMul_yy : muMul_yy;

            mu_SxH.y = m_yy * SxH.y;

            float m_zz = (mu_zz != NULL) ? mu_zz[x0] * muMul_zz : muMul_zz;

            mu_SxH.z = m_zz * SxH.z;

            float3 _Sxmu_SxH = crossf(mu_SxH, S);

            tx[x0] = _Sxmu_SxH.x;
            ty[x0] = _Sxmu_SxH.y;
            tz[x0] = _Sxmu_SxH.z;
        }
    }

    __export__  void baryakhtar_transverse_async(float** tx, float**  ty, float**  tz,
            float**  Sx, float**  Sy, float**  Sz,
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
                    Sx[dev], Sy[dev], Sz[dev],
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
