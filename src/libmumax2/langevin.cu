#include "langevin.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__device__ float findroot(func* f, float prefix, float mult, float xa, float xb) {

    float ya = f[0](xa, prefix, mult);
    if (fabsf(ya) < eps) return xa;
    float yb = f[0](xb, prefix, mult);
    if (fabsf(yb) < eps) return xb;
    
    float y1 = ya;
    float x1 = xa;
    float y2 = yb;
    float x2 = xb;
    
    float x = 1.0e10f;
    float y = 1.0e10f;
    float ty = y;
    
    float teps = y;
    
    while (teps > eps) {
    
        float k = (x2 - x1) / (y2 - y1);
        x = x1 - y1 * k;
        
        y = f[0](x, prefix, mult);
        
        y1 = (signbit(y) == signbit(y1)) ? y : y1;
        x1 = (signbit(y) == signbit(y1)) ? x : x1;
        y2 = (signbit(y) == signbit(y2) && x1 != x) ? y : y2;
        x2 = (signbit(y) == signbit(y2) && x1 != x) ? x : x2;
        
        teps = fabsf(y - ty);
        ty = y;
    }
    
    return x;
}

__device__ float Model(float m, float prefix, float pre) {
            float x = pre * m;
            float val = prefix * L(x) - m;
            //printf("L(%g) - %g = %g\n", x, m, val);
            return val;
}

__device__ func pModel = Model;

__global__ void langevinKern(float* __restrict__ msat0Msk,
                             float* __restrict__ msat0T0Msk,
                              float* __restrict__ T,
                              float* __restrict__ J0Msk,
                              const float msat0Mul,
                              const float msat0T0Mul,
                              const float J0Mul,
                              int Npart) {
            int i = threadindex;
	        if (i < Npart) {
	            float Temp = T[i];
	            float msat0T0 = (msat0T0Msk == NULL) ? msat0T0Mul : msat0T0Mul * msat0T0Msk[i];
	            
	            if (Temp == 0.0f) {
	                   msat0Msk[i] = msat0T0 / msat0Mul;
	                   return;
	            }
	            
	            float J0 = (J0Msk == NULL) ? J0Mul : J0Mul * J0Msk[i];
	            float pre = msat0Mul * J0 / (kB * Temp);
	            float prefix = (msat0T0 / msat0Mul);
	            float msat0 =  findroot(&pModel, prefix, pre, 0.9f, 1.5f);
	            //if (msat0 < 0.0f) { printf("msat: %f\n", msat0); }
	            msat0Msk[i] = fabsf(msat0);
	        }
}

__export__ void langevinAsync(float** msat0, 
                              float** msat0T0,
                              float** T, 
                              float** J0,
                              const float msat0Mul,
                              const float msat0T0Mul,
                              const float J0Mul,
                              int Npart,
                              CUstream* stream) {
            dim3 gridSize, blockSize;
	        make1dconf(Npart, &gridSize, &blockSize);
	        for (int dev = 0; dev < nDevice(); dev++) {
		        assert(msat0[dev] != NULL);
		        assert(T[dev] != NULL);
		        gpu_safe(cudaSetDevice(deviceId(dev)));
		        langevinKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (msat0[dev],
		                                                                              msat0T0[dev],
		                                                                               T[dev],
		                                                                               J0[dev],
		                                                                               msat0Mul,
		                                                                               msat0T0Mul,
		                                                                               J0Mul,
		                                                                               Npart);
	        }
                              
}

#ifdef __cplusplus
}
#endif
