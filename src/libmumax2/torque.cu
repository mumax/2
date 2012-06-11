#include "torque.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void torqueKern(float* tx, float* ty, float* tz,
                           float* mx, float* my, float* mz, 
                           float* hx, float* hy, float* hz, 
						   float* alpha_map, float alpha_mul,
						   int Npart) {
	int i = threadindex;
	if (i < Npart) {
		// reconstruct alpha from map+multiplier
		float alpha;
		if(alpha_map == NULL){
			alpha = alpha_mul;
		}else{
			alpha = alpha_mul * alpha_map[i];
		}

    	float Mx = mx[i];
    	float My = my[i];
    	float Mz = mz[i];
    	
    	float Hx = hx[i];
    	float Hy = hy[i];
    	float Hz = hz[i];
    	
    	//  m cross H
    	float _mxHx =  My * Hz - Hy * Mz;
    	float _mxHy = -Mx * Hz + Hx * Mz;
    	float _mxHz =  Mx * Hy - Hx * My;

    	// - m cross (m cross H)
    	float _mxmxHx = -My * _mxHz + _mxHy * Mz;
    	float _mxmxHy = +Mx * _mxHz - _mxHx * Mz;
    	float _mxmxHz = -Mx * _mxHy + _mxHx * My;

		float gilb = 1.0f / (1.0f + alpha * alpha);
    	tx[i] = gilb * (_mxHx + _mxmxHx * alpha);
    	ty[i] = gilb * (_mxHy + _mxmxHy * alpha);
    	tz[i] = gilb * (_mxHz + _mxmxHz * alpha);
	}
}


__export__ void torqueAsync(float** tx, float** ty, float** tz, float** mx, float** my, float** mz, float** hx, float** hy, float** hz, float** alpha_map, float alpha_mul, CUstream* stream, int Npart) {
	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(tx[dev] != NULL);
		assert(ty[dev] != NULL);
		assert(tz[dev] != NULL);
		assert(mx[dev] != NULL);
		assert(my[dev] != NULL);
		assert(mz[dev] != NULL);
		assert(hx[dev] != NULL);
		assert(hy[dev] != NULL);
		assert(hz[dev] != NULL);
		// alphaMap may be null
		gpu_safe(cudaSetDevice(deviceId(dev)));
		torqueKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (tx[dev],ty[dev],tz[dev],  mx[dev],my[dev],mz[dev], hx[dev],hy[dev],hz[dev], alpha_map[dev], alpha_mul, Npart);
	}
}

#ifdef __cplusplus
}
#endif
