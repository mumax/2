#include "exchange6.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

/// 2D, plane per plane, i=plane index
    __global__ void exchange6Kern(float* hx, float* hy, float* hz,
                                  float* mx, float* my, float* mz,
                                  float* mSat_map, float* Aex_map, float Aex2_Mu0Msat_mul,
                                  int N0, int N1, int N2,
                                  int wrap0, int wrap1, int wrap2,
                                  float cellx_2, float celly_2, float cellz_2)
    {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;

        int I = i * N1 * N2 + j * N2 + k;

        if (i < N0 && j < N1 && k < N2)
        {

            float mSat_mask = (mSat_map == NULL) ? 1.0f : mSat_map[I];
            mSat_mask = (mSat_mask == 0.0f) ? 1.0f : mSat_mask;


            float Aex2_Mu0Msat = (Aex_map == NULL) ? Aex2_Mu0Msat_mul / mSat_mask : (Aex2_Mu0Msat_mul / mSat_mask) * Aex_map[I]; // 2 * Aex / Mu0 * Msat

            float mx0 = mx[I]; // mag component of central cell
            float mx1, mx2 ;   // mag component of neighbors in 2 directions

            float my0 = my[I]; // mag component of central cell
            float my1, my2 ;   // mag component of neighbors in 2 directions

            float mz0 = mz[I]; // mag component of central cell
            float mz1, mz2 ;   // mag component of neighbors in 2 directions

            float Hx, Hy, Hz;

            // neighbors in X direction
            int idx = i - 1;
            idx = (idx < 0 && wrap0) ? N0 - 1 : idx;
            mx1 = (idx < 0) ? 0.0f : mx[idx * N1 * N2 + j * N2 + k];
            my1 = (idx < 0) ? 0.0f : my[idx * N1 * N2 + j * N2 + k];
            mz1 = (idx < 0) ? 0.0f : mz[idx * N1 * N2 + j * N2 + k];

            idx = i + 1;
            idx = (idx == N0 && wrap0) ? 0 : idx;
            mx2 = (idx == N0) ? 0.0f : mx[idx * N1 * N2 + j * N2 + k];
            my2 = (idx == N0) ? 0.0f : my[idx * N1 * N2 + j * N2 + k];
            mz2 = (idx == N0) ? 0.0f : mz[idx * N1 * N2 + j * N2 + k];

            Hx = Aex2_Mu0Msat * cellx_2 * ((mx1 - mx0) + (mx2 - mx0));
            Hy = Aex2_Mu0Msat * cellx_2 * ((my1 - my0) + (my2 - my0));
            Hz = Aex2_Mu0Msat * cellx_2 * ((mz1 - mz0) + (mz2 - mz0));

            // neighbors in Z direction
            idx = k - 1;
            idx = (idx < 0 && wrap2) ? N2 - 1 : idx;
            mx1 = (idx < 0) ? 0.0f : mx[i * N1 * N2 + j * N2 + idx];
            my1 = (idx < 0) ? 0.0f : my[i * N1 * N2 + j * N2 + idx];
            mz1 = (idx < 0) ? 0.0f : mz[i * N1 * N2 + j * N2 + idx];

            idx = k + 1;
            idx = (idx == N2 && wrap2) ? 0 : idx;
            mx2 = (idx == N2) ? 0.0f : mx[i * N1 * N2 + j * N2 + idx];
            my2 = (idx == N2) ? 0.0f : my[i * N1 * N2 + j * N2 + idx];
            mz2 = (idx == N2) ? 0.0f : mz[i * N1 * N2 + j * N2 + idx];

            Hx += Aex2_Mu0Msat * cellz_2 * ((mx1 - mx0) + (mx2 - mx0));
            Hy += Aex2_Mu0Msat * cellz_2 * ((my1 - my0) + (my2 - my0));
            Hz += Aex2_Mu0Msat * cellz_2 * ((mz1 - mz0) + (mz2 - mz0));
            
            // neighbors in Y direction
            idx = j - 1;
            idx = (idx < 0 && wrap1) ? N1 - 1 : idx;
            mx1 = (idx < 0) ? 0.0f : mx[i * N1 * N2 + idx * N2 + k];
            my1 = (idx < 0) ? 0.0f : my[i * N1 * N2 + idx * N2 + k];
            mz1 = (idx < 0) ? 0.0f : mz[i * N1 * N2 + idx * N2 + k];

            idx = j + 1;
            idx = (idx == N1 && wrap1) ? 0 : idx;
            mx2 = (idx == N1) ? 0.0f : mx[i * N1 * N2 + idx * N2 + k];
            my2 = (idx == N1) ? 0.0f : my[i * N1 * N2 + idx * N2 + k];
            mz2 = (idx == N1) ? 0.0f : mz[i * N1 * N2 + idx * N2 + k];

            Hx += Aex2_Mu0Msat * celly_2 * ((mx1 - mx0) + (mx2 - mx0));
            Hy += Aex2_Mu0Msat * celly_2 * ((my1 - my0) + (my2 - my0));
            Hz += Aex2_Mu0Msat * celly_2 * ((mz1 - mz0) + (mz2 - mz0));

            // Write back to global memory
            hx[I] = Hx;
            hy[I] = Hy;
            hz[I] = Hz;

        }

    }


#define BLOCKSIZEX 4
#define BLOCKSIZEY 4
#define BLOCKSIZEZ 32

    __export__ void exchange6Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float** msat, float** aex, float Aex2_mu0MsatMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams)
    {

        dim3 gridsize(divUp(N0, BLOCKSIZEX), divUp(N1Part, BLOCKSIZEY), divUp(N2, BLOCKSIZEZ));
        dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);

        //int NPart = N0 * N1Part * N2;

        float cellx_2 = 1.0f / (cellSizeX * cellSizeX);
        float celly_2 = 1.0f / (cellSizeY * cellSizeY);
        float cellz_2 = 1.0f / (cellSizeZ * cellSizeZ);
        //printf("exchange factors %g %g %g\n", fac0, fac1, fac2); // OK

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

