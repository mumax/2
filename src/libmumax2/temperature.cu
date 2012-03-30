
#include "temperature.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void temperature_scaleKern(float* noise, 
				float* alphaMask,
			   	float* tempMask, float alphaKB2tempMul,
			   	float* mSatMask, float mu0VgammaDtMSatMul,
			   	int Npart){


	int i = threadindex;
	if (i < Npart) {

		float alphaMul;
		if(alphaMask != NULL){
			alphaMul = alphaMask[i];
		}else{
			alphaMul = 1.0f;
		}

		float mSatMul;
		if(mSatMask != NULL){
			mSatMul = mSatMask[i];
		}else{
			mSatMul = 1.0f;
		}

		float tempMul;
		if(tempMask != NULL){
			tempMul = tempMask[i];
		}else{
			tempMul = 1.0f;
		}

		if(mSatMul != 0.f){
			noise[i] *= sqrtf((alphaMul * tempMul * alphaKB2tempMul)/(mu0VgammaDtMSatMul * mSatMul));
		}else{
			// no fluctuations outside magnet
			noise[i] = 0.f;
		}
	}
}


__export__ void temperature_scaleNoise(float** noise,
			   	float** alphaMask, 
			   	float** tempMask, float alphaKB2tempMul,
			   	float** mSatMask, 
			   	float mu0VgammaDtMSatMul,
			   	CUstream* stream, int Npart){

	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		temperature_scaleKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (
						noise[dev], alphaMask[dev], tempMask[dev], alphaKB2tempMul, mSatMask[dev], mu0VgammaDtMSatMul, Npart);
	}
}

#ifdef __cplusplus
}
#endif
