#include "brillouin.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__device__ float findroot_Ridders_Ts(funcTs* f, float J, float mult, float C, float xa, float xb)
{

    float ya = f[0](xa, J, mult, C);
    if (fabsf(ya) < zero) return xa;
    float yb = f[0](xb, J, mult, C);
    if (fabsf(yb) < zero) return xb;

    float y1 = ya;
    float x1 = xa;
    float y2 = yb;
    float x2 = xb;

    float x = 1.0e10f;
    float y = 1.0e10f;
    float tx = x;

    float teps = x;

    float x3 = 0.0f;
    float y3 = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    int iter = 0;
    while (teps > eps && iter < 1000)
    {

        x3 = 0.5f * (x2 + x1);
        y3 = f[0](x3, J, mult, C);

        dy = (y3 * y3 - y1 * y2);
        if (dy == 0.0f)
        {
            x = x3;
            break;
        }

        dx = (x3 - x1) * signf(y1 - y2) * y3 / (sqrtf(dy));

        x = x3 + dx;
        y = f[0](x, J, mult, C);

        y2 = (signbit(y) == signbit(y3)) ? y2 : y3;
        x2 = (signbit(y) == signbit(y3)) ? x2 : x3;

        y2 = (signbit(y) == signbit(y1) || x2 == x3) ? y2 : y1;
        x2 = (signbit(y) == signbit(y1) || x2 == x3) ? x2 : x1;

        y1 = y;
        x1 = x;

        teps = fabsf((x - tx) / (tx + x));

        tx = x;
        iter++;

    }
    return x;
}

// here n = <Sz>/ S
// <Sz> = n * S
// <Sz> = S * Bj(S*J0*<Sz>/(kT))

__device__ float ModelTs(float n, float J, float pre, float C)
{
    float x = (n == 0.0f) ? 1.0e38f : pre / n;
    float val = Bj(J, x) - C;
    return val;
}

__device__ funcTs pModelTs = ModelTs;

__global__ void tsKern(float* __restrict__ Ts,
                      float* __restrict__ msatMsk,
                      float* __restrict__ msat0T0Msk,
                      float* __restrict__ TcMsk,
                      float* __restrict__ SMsk,
                      const float msatMul,
                      const float msat0T0Mul,
                      const float TcMul,
                      const float SMul,
                      int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        float msat0T0 = msat0T0Mul * getMaskUnity(msat0T0Msk, i);
        if (msat0T0 == 0.0f)
        {
            Ts[i] = 0.0f;
            return;
        }

        float msat = msatMul * getMaskUnity(msatMsk, i);
        if (msat == msat0T0) {
        	Ts[i] = 0.0f;
        	return;
        }

        float Tc = TcMul * getMaskUnity(TcMsk, i);
        if (msat == 0.0f)
        {
            Ts[i] = Tc;
            return;
        }

        float S  = (SMsk  == NULL) ? SMul  : SMul  * SMsk[i];

        float J0  = 3.0f * Tc / (S * (S + 1.0f));
        float m = msat / msat0T0;
        float pre = S * S * J0 * m;
        float T = findroot_Ridders_Ts(&pModelTs, S, pre, m, 0.0f, Tc);

        Ts[i] = T;
    }
}

__export__ void tsAsync(float** Ts,
                              float** msat,
                              float** msat0T0,
                              float** Tc,
                              float** S,
                              const float msatMul,
                              const float msat0T0Mul,
                              const float TcMul,
                              const float SMul,
                              int Npart,
                              CUstream* stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    int dev = 0;
    gpu_safe(cudaSetDevice(deviceId(dev)));
    tsKern <<< gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (Ts[dev],
            msat[dev],
            msat0T0[dev],
            Tc[dev],
            S[dev],
            msatMul,
            msat0T0Mul,
            TcMul,
            SMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
