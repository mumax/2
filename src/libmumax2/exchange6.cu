#include "exchange6.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif
// full 3D blocks
__global__ void exchange6Kern(float* __restrict__ hx, float* __restrict__  hy, float* __restrict__  hz,
                              float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                              float* __restrict__  mSat_map, float* __restrict__  Aex_map,
                              const float Aex2_Mu0Msat_mul,
                              const int N0, const int N1, const int N2,
                              const int wrap0, const int wrap1, const int wrap2,
                              const float cellx_2, const float celly_2, const float cellz_2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N0 && j < N1 && k < N2)
    {

        int I = i * N1 * N2 + j * N2 + k;

        float mSat0 = getMaskUnity(mSat_map, I);
        float Aex0 = getMaskUnity(Aex_map, I);
        float lex0Mul = fdivZero(Aex0, mSat0);
        float lexMul, lex1Mul, lex2Mul;
        
        float pre_x = cellx_2 * Aex2_Mu0Msat_mul;
        float pre_y = celly_2 * Aex2_Mu0Msat_mul;
        float pre_z = cellz_2 * Aex2_Mu0Msat_mul;
        
        float mx0 = mx[I]; // mag component of central cell
        float mx1, mx2 ;   // mag component of neighbors in 2 directions

        float my0 = my[I]; // mag component of central cell
        float my1, my2 ;   // mag component of neighbors in 2 directions

        float mz0 = mz[I]; // mag component of central cell
        float mz1, mz2 ;   // mag component of neighbors in 2 directions

        float Hx, Hy, Hz;
        float tHx, tHy, tHz;

        // neighbors in X direction
        int idx = i - 1;
        idx = (idx < 0 && wrap0) ? N0 + idx : idx;
        lex1Mul = (idx < 0) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx1 = (idx < 0) ? mx0 : mx[idx * N1 * N2 + j * N2 + k];
        my1 = (idx < 0) ? my0 : my[idx * N1 * N2 + j * N2 + k];
        mz1 = (idx < 0) ? mz0 : mz[idx * N1 * N2 + j * N2 + k];
        lexMul = 2.0f * fdivZero((lex0Mul * lex1Mul), (lex0Mul + lex1Mul));
        tHx = lexMul * (mx1 - mx0);
        tHy = lexMul * (my1 - my0);
        tHz = lexMul * (mz1 - mz0);

        idx = i + 1;
        idx = (idx == N0 && wrap0) ? idx - N0 : idx;
        lex2Mul = (idx == N0) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx2 = (idx == N0) ? mx0 : mx[idx * N1 * N2 + j * N2 + k];
        my2 = (idx == N0) ? my0 : my[idx * N1 * N2 + j * N2 + k];
        mz2 = (idx == N0) ? mz0 : mz[idx * N1 * N2 + j * N2 + k];
        lexMul = 2.0f * fdivZero((lex0Mul * lex2Mul), (lex0Mul + lex2Mul));
        tHx += lexMul * (mx2 - mx0);
        tHy += lexMul * (my2 - my0);
        tHz += lexMul * (mz2 - mz0);

        Hx = pre_x * tHx;
        Hy = pre_x * tHy;
        Hz = pre_x * tHz;
        
        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && wrap2) ? N2 + idx : idx;
        lex1Mul = (idx < 0) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx1 = (idx < 0) ? mx0 : mx[i * N1 * N2 + j * N2 + idx];
        my1 = (idx < 0) ? my0 : my[i * N1 * N2 + j * N2 + idx];
        mz1 = (idx < 0) ? mz0 : mz[i * N1 * N2 + j * N2 + idx];
        lexMul = 2.0f * fdivZero((lex0Mul * lex1Mul), (lex0Mul + lex1Mul));
        tHx = lexMul * (mx1 - mx0);
        tHy = lexMul * (my1 - my0);
        tHz = lexMul * (mz1 - mz0);

        idx = k + 1;
        idx = (idx == N2 && wrap2) ? idx - N2 : idx;
        lex2Mul = (idx == N2) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx2 = (idx == N2) ? mx0 : mx[i * N1 * N2 + j * N2 + idx];
        my2 = (idx == N2) ? my0 : my[i * N1 * N2 + j * N2 + idx];
        mz2 = (idx == N2) ? mz0 : mz[i * N1 * N2 + j * N2 + idx];
        lexMul = 2.0f * fdivZero((lex0Mul * lex2Mul), (lex0Mul + lex2Mul));
        tHx += lexMul * (mx2 - mx0);
        tHy += lexMul * (my2 - my0);
        tHz += lexMul * (mz2 - mz0);

        Hx += pre_z * tHx;
        Hy += pre_z * tHy;
        Hz += pre_z * tHz;
        
        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && wrap1) ? N1 + idx : idx;
        lex1Mul = (idx < 0) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx1 = (idx < 0) ? mx0 : mx[i * N1 * N2 + idx * N2 + k];
        my1 = (idx < 0) ? my0 : my[i * N1 * N2 + idx * N2 + k];
        mz1 = (idx < 0) ? mz0 : mz[i * N1 * N2 + idx * N2 + k];
        lexMul = 2.0f * fdivZero((lex0Mul * lex1Mul), (lex0Mul + lex1Mul));
        tHx = lexMul * (mx1 - mx0);
        tHy = lexMul * (my1 - my0);
        tHz = lexMul * (mz1 - mz0);

        idx = j + 1;
        idx = (idx == N1 && wrap1) ? idx - N1 : idx;
        lex2Mul = (idx == N1) ? 0.0f : fdivZero(getMaskUnity(Aex_map, idx), getMaskUnity(mSat_map, idx));
        mx2 = (idx == N1) ? mx0 : mx[i * N1 * N2 + idx * N2 + k];
        my2 = (idx == N1) ? my0 : my[i * N1 * N2 + idx * N2 + k];
        mz2 = (idx == N1) ? mz0 : mz[i * N1 * N2 + idx * N2 + k];
        lexMul = 2.0f * fdivZero((lex0Mul * lex2Mul), (lex0Mul + lex2Mul));
        tHx += lexMul * (mx2 - mx0);
        tHy += lexMul * (my2 - my0);
        tHz += lexMul * (mz2 - mz0);
        
        Hx += pre_y * tHx;
        Hy += pre_y * tHy;
        Hz += pre_y * tHz;

        // Write back to global memory
        hx[I] = Hx;
        hy[I] = Hy;
        hz[I] = Hz;

    }

}


__export__ void exchange6Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float** msat, float** aex, float Aex2_mu0MsatMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams)
{

    dim3 gridsize, blocksize;

    make3dconf(N0, N1Part, N2, &gridsize, &blocksize);

    float cellx_2 = (float)(1.0 / ((double)cellSizeX * (double)cellSizeX));
    float celly_2 = (float)(1.0 / ((double)cellSizeY * (double)cellSizeY));
    float cellz_2 = (float)(1.0 / ((double)cellSizeZ * (double)cellSizeZ));

    int nDev = nDevice();

    for (int dev = 0; dev < nDev; dev++)
    {
        gpu_safe(cudaSetDevice(deviceId(dev)));
        exchange6Kern <<< gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(hx[dev], hy[dev], hz[dev], mx[dev], my[dev], mz[dev], msat[dev], aex[dev], Aex2_mu0MsatMul, N0, N1Part, N2, periodic0, periodic1, periodic2, cellx_2, celly_2, cellz_2);
    }
}


#ifdef __cplusplus
}
#endif

