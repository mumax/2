#include "baryakhtar-non-local.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

    __global__ void llbarNonlocal02ncKern(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,

            float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
            float* __restrict__ lhx, float* __restrict__ lhy, float* __restrict__ lhz,
            float* __restrict__ rhx, float* __restrict__ rhy, float* __restrict__ rhz,

            float* __restrict__ msat0T0Msk,

            float* __restrict__ lambda_e_xx,
            float* __restrict__ lambda_e_yy,
            float* __restrict__ lambda_e_zz,

            const float lambda_eMul_xx,
            const float lambda_eMul_yy,
            const float lambda_eMul_zz,

            const int4 size,
            const float3 mstep,
            const int3 pbc,
            const int i)
    {

        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int k = blockIdx.y * blockDim.y + threadIdx.y;

        if (j < size.y && k < size.z)
        {

            int x0 = i * size.w + j * size.z + k;

            float msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];
            // make sure there is no torque in vacuum!
            if (msat0T0 == 0.0f)
            {
                tx[x0] = 0.0f;
                ty[x0] = 0.0f;
                tz[x0] = 0.0f;
                return;
            }

            float3 mmstep = make_float3(mstep.x, mstep.y, mstep.z);

            // Second-order derivative 3-points stencil
//==================================================================================================

            int xb1, xf1, x;
            xb1 = (i == 0 && pbc.x == 0) ? i     : i - 1;
            x   = (i == 0 && pbc.x == 0) ? i + 1 : i;
            xf1 = (i == 0 && pbc.x == 0) ? i + 2 : i + 1;
            xb1 = (i == size.x - 1 && pbc.x == 0) ? i - 2 : xb1;
            x   = (i == size.x - 1 && pbc.x == 0) ? i - 1 : x;
            xf1 = (i == size.x - 1 && pbc.x == 0) ? i     : xf1;

            int yb1, yf1, y;
            yb1 = (j == 0 && lhx == NULL) ? j     : j - 1;
            y   = (j == 0 && lhx == NULL) ? j + 1 : j;
            yf1 = (j == 0 && lhx == NULL) ? j + 2 : j + 1;
            yb1 = (j == size.y - 1 && rhx == NULL) ? j - 2 : yb1;
            y   = (j == size.y - 1 && rhx == NULL) ? j - 1 : y;
            yf1 = (j == size.y - 1 && rhx == NULL) ? j     : yf1;

            int zb1, zf1, z;
            zb1 = (k == 0 && pbc.z == 0) ? k     : k - 1;
            z   = (k == 0 && pbc.z == 0) ? k + 1 : k;
            zf1 = (k == 0 && pbc.z == 0) ? k + 2 : k + 1;
            zb1 = (k == size.z - 1 && pbc.z == 0) ? k - 2 : zb1;
            z   = (k == size.z - 1 && pbc.z == 0) ? k - 1 : z;
            zf1 = (k == size.z - 1 && pbc.z == 0) ? k     : zf1;

            xb1 = (xb1 < 0) ?          size.x + xb1 : xb1;
            xf1 = (xf1 > size.x - 1) ? xf1 - size.x : xf1;

            yb1 = (yb1 < 0) ?          size.y + yb1 : yb1;
            yf1 = (yf1 > size.y - 1) ? yf1 - size.y : yf1;

            zb1 = (zb1 < 0) ?          size.z + zb1 : zb1;
            zf1 = (zf1 > size.z - 1) ? zf1 - size.z : zf1;

            int comm = j * size.z + k;
            int3 xn = make_int3(xb1 * size.w + comm,
                                x   * size.w + comm,
                                xf1 * size.w + comm);


            comm = i * size.w + k;
            int3 yn = make_int3(yb1 * size.z + comm,
                                y   * size.z + comm,
                                yf1 * size.z + comm);


            comm = i * size.w + j * size.z;
            int3 zn = make_int3(zb1 + comm,
                                z   + comm,
                                zf1 + comm);


            float h_b1, h, h_f1;
            float ddhx_x, ddhx_y, ddhx_z;
            float ddhy_x, ddhy_y, ddhy_z;
            float ddhz_x, ddhz_y, ddhz_z;

            float ddhx, ddhy, ddhz;
            float sum;

            h_b1   = hx[xn.x];
            h      = hx[xn.y];
            h_f1   = hx[xn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhx_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = (j > 0 || lhx == NULL) ? hx[yn.x] : lhx[yn.x];
            h    = hx[yn.y];
            h_f1 = (j < size.y - 1 || rhx == NULL) ? hx[yn.z] : rhx[yn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhx_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = hx[zn.x];
            h    = hx[zn.y];
            h_f1 = hx[zn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhx_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            ddhx   = mmstep.x * ddhx_x + mmstep.y * ddhx_y + mmstep.z * ddhx_z;

            h_b1   = hy[xn.x];
            h      = hy[xn.y];
            h_f1   = hy[xn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhy_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = (j > 0 || lhx == NULL) ? hy[yn.x] : lhy[yn.x];
            h    = hy[yn.y];
            h_f1 = (j < size.y - 1 || rhx == NULL) ? hy[yn.z] : rhy[yn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhy_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = hy[zn.x];
            h    = hy[zn.y];
            h_f1 = hy[zn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhy_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            ddhy   = mmstep.x * ddhy_x + mmstep.y * ddhy_y + mmstep.z * ddhy_z;

            h_b1   = hz[xn.x];
            h      = hz[xn.y];
            h_f1   = hz[xn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhz_x = (size.x > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = (j > 0 || lhx == NULL) ? hz[yn.x] : lhz[yn.x];
            h    = hz[yn.y];
            h_f1 = (j < size.y - 1 || rhx == NULL) ? hz[yn.z] : rhz[yn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhz_y = (size.y > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            h_b1 = hz[zn.x];
            h    = hz[zn.y];
            h_f1 = hz[zn.z];
            sum    = __fadd_rn(h_b1, h_f1);
            ddhz_z = (size.z > 2) ? __fmaf_rn(-2.0f, h, sum) : 0.0;

            ddhz   = mmstep.x * ddhz_x + mmstep.y * ddhz_y + mmstep.z * ddhz_z;
//==================================================================================================

            float le_xx = (lambda_e_xx != NULL) ? lambda_e_xx[x0] * lambda_eMul_xx : lambda_eMul_xx;
            float ledHx = le_xx * ddhx;

            float le_yy = (lambda_e_yy != NULL) ? lambda_e_yy[x0] * lambda_eMul_yy : lambda_eMul_yy;

            float ledHy = le_yy * ddhy;

            float le_zz = (lambda_e_zz != NULL) ? lambda_e_zz[x0] * lambda_eMul_zz : lambda_eMul_zz;
            float ledHz = le_zz * ddhz;

            tx[x0] = -ledHx;
            ty[x0] = -ledHy;
            tz[x0] = -ledHz;
        }
    }

#define BLOCKSIZE 16

    __export__  void llbar_nonlocal02nc_async(float** tx, float**  ty, float**  tz,
            float**  hx, float**  hy, float**  hz,

            float** msat0T0,

            float** lambda_e_xx,
            float** lambda_e_yy,
            float** lambda_e_zz,

            const float lambda_eMul_xx,
            const float lambda_eMul_yy,
            const float lambda_eMul_zz,

            const int sx, const int sy, const int sz,
            const float csx, const float csy, const float csz,
            const int pbc_x, const int pbc_y, const int pbc_z,
            CUstream* stream)
    {

        dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
        dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

        float icsx2 = 1.0f / (csx * csx);
        float icsy2 = 1.0f / (csy * csy);
        float icsz2 = 1.0f / (csz * csz);

        int syz = sy * sz;


        float3 mstep = make_float3(icsx2, icsy2, icsz2);
        int4 size = make_int4(sx, sy, sz, syz);
        int3 pbc = make_int3(pbc_x, pbc_y, pbc_z);

        int nDev = nDevice();


        for (int dev = 0; dev < nDev; dev++)
        {
            gpu_safe(cudaSetDevice(deviceId(dev)));

            // calculate dev neighbours

            int ld = Mod(dev - 1, nDev);
            int rd = Mod(dev + 1, nDev);

            float* lhx = hx[ld];
            float* lhy = hy[ld];
            float* lhz = hz[ld];

            float* rhx = hx[rd];
            float* rhy = hy[rd];
            float* rhz = hz[rd];

            if(pbc_y == 0)
            {
                if(dev == 0)
                {
                    lhx = NULL;
                    lhy = NULL;
                    lhz = NULL;
                }
                if(dev == nDev - 1)
                {
                    rhx = NULL;
                    rhy = NULL;
                    rhz = NULL;
                }
            }


            for (int i = 0; i < sx; i++)
            {

                llbarNonlocal02ncKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev], ty[dev], tz[dev],
                        hx[dev], hy[dev], hz[dev],
                        lhx, lhy, lhz,
                        rhx, rhy, rhz,

                        msat0T0[dev],

                        lambda_e_xx[dev],
                        lambda_e_yy[dev],
                        lambda_e_zz[dev],

                        lambda_eMul_xx,
                        lambda_eMul_yy,
                        lambda_eMul_zz,

                        size,
                        mstep,
                        pbc,
                        i);
            }

        } // end dev < nDev loop

    }

    // ========================================

#ifdef __cplusplus
}
#endif
