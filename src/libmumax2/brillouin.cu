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
__device__ float findroot_Ridders(func* f, float J, float mult, float xa, float xb) {

    float ya = f[0](xa, J, mult);
    if (fabsf(ya) < zero) return xa;
    float yb = f[0](xb, J, mult);
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
    while (teps > eps && iter < 1000) {
	
	x3 = 0.5f * (x2 + x1);
	y3 = f[0](x3, J, mult);
	
	dy = (y3*y3 - y1*y2);
	if (dy == 0.0f) {
	    x = x3;
	    break;
	}
	
	dx = (x3 - x1) * signf(y1 - y2) * y3 / (sqrtf(dy)); 
	
        x = x3 + dx;
        y = f[0](x, J, mult);
	
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

///@internal
__device__ float findroot_Secant(func* f, float J, float mult, float xa, float xb) {

    float ya = f[0](xa, J, mult);
    if (fabsf(ya) < zero) return xa;
    float yb = f[0](xb, J, mult);
    if (fabsf(yb) < zero) return xb;
    
    float y1 = ya;
    float x1 = xa;
    float y2 = yb;
    float x2 = xb;
    
    float x = 1.0e10f;
    float y = 1.0e10f;
    float tx = x;
    
    float teps = x;
    
    int iter = 0;
    //int i = threadindex;
    while (teps > eps && iter < 100000) {
	
        float k = (x2 - x1) / (y2 - y1);
        x = x1 - y1 * k;
        
        y = f[0](x, J, mult);
        
        y1 = (signbit(y) == signbit(y1)) ? y : y1;
        x1 = (signbit(y) == signbit(y1)) ? x : x1;
        y2 = (signbit(y) == signbit(y2) && x1 != x) ? y : y2;
        x2 = (signbit(y) == signbit(y2) && x1 != x) ? x : x2;
        
        teps = fabsf((x - tx) / (tx + x));
	    
        tx = x;
	iter++;
    }
    // ~ if (i == 0) {
	// ~ printf("Total number of iterations: %d\n", iter);
    // ~ }
    return x;
}

// here n = m / me
// <Sz> = n * J
// <Sz> = J * Bj(S*J0*<Sz>/(kT))

__device__ float Model(float n, float J, float pre) {
            float x = pre * n;
            float val = Bj(J, x) - n;
            //printf("B(%g) - %g = %g\n", x, n, val);
            return val;
}

__device__ func pModel = Model;

__global__ void brillouinKern(float* __restrict__ msat0Msk,
                             float* __restrict__ msat0T0Msk,
                              float* __restrict__ T,
                              float* __restrict__ TcMsk,
                              float* __restrict__ SMsk,
                              const float msat0Mul,
                              const float msat0T0Mul,
                              const float TcMul,
                              const float SMul,
                              int Npart) {
            int i = threadindex;
	        if (i < Npart) {
	            float Temp = T[i];
	            
	            float msat0T0 = (msat0T0Msk == NULL) ? msat0T0Mul : msat0T0Mul * msat0T0Msk[i];
	            
	            if (Temp == 0.0f) {
	                   msat0Msk[i] = msat0T0 / msat0Mul;
	                   return;
	            }
		    
	            float Tc = (TcMsk == NULL) ? TcMul : TcMul * TcMsk[i];
		    
		    if (Temp > Tc) {
			msat0Msk[i] = 0.0f;
			return;
		    }
		    
	            float S  = (SMsk  == NULL) ? SMul  : SMul  * SMsk[i];
	            
	            float J0  = 3.0f * Tc / (S * (S + 1.0f));
	            float pre = S * S * J0 / (Temp);
		    
		    float dT = Tc - Temp;
		    float lowLimit = (dT < 0.25f) ? -0.1f : 0.1f;
		    float hiLimit  = (dT < 0.25f) ?  0.5f : 1.1f; 
	            float msat0 = findroot_Ridders(&pModel, S, pre, lowLimit, hiLimit);
		    
	            msat0Msk[i] = msat0T0 * fabsf(msat0) / (msat0Mul);
	        }
}

__export__ void brillouinAsync(float** msat0, 
                              float** msat0T0,
                              float** T, 
                              float** Tc,
                              float** S,
                              const float msat0Mul,
                              const float msat0T0Mul,
                              const float TcMul,
                              const float SMul,
                              int Npart,
                              CUstream* stream) {
            dim3 gridSize, blockSize;
	        make1dconf(Npart, &gridSize, &blockSize);
	        for (int dev = 0; dev < nDevice(); dev++) {
		        assert(msat0[dev] != NULL);
		        assert(T[dev] != NULL);
		        gpu_safe(cudaSetDevice(deviceId(dev)));
		        brillouinKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (msat0[dev],
		                                                                              msat0T0[dev],
		                                                                               T[dev],
		                                                                               Tc[dev],
		                                                                               S[dev],
		                                                                               msat0Mul,
		                                                                               msat0T0Mul,
		                                                                               TcMul,
		                                                                               SMul,
		                                                                               Npart);
	        }
                              
}

#ifdef __cplusplus
}
#endif
