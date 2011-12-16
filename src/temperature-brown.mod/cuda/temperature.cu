
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
				float* alphaMap, float alphaMul,
			   	float* tempMap, float kB2tempMul,
			   	float* mSatMap, float mSatMul,
			   	float mu0VgammaDt,
			   	int Npart){


	int i = threadindex;
	if (i < Npart) {

		float alpha;
		if(alphaMap != NULL){
			alpha = alphaMap[i] * alphaMul;
		}else{
			alpha = alphaMul;
		}

		float mSat;
		if(mSatMap != NULL){
			mSat = mSatMap[i] * mSatMul;
		}else{
			mSat = mSatMul;
		}

		float kB2temp;
		if(tempMap != NULL){
			kB2temp = tempMap[i] * kB2tempMul;
		}else{
			kB2temp = kB2tempMul;
		}

		
		noise[i] *= sqrtf((alpha * kB2temp)/(mu0VgammaDt * mSat));
	}
}


void temperature_scaleNoise(float** noise,
			   	float** alpha, float alphaMul,
			   	float** temp, float kB2tempMul,
			   	float** mSat, float msatMul,
			   	float mu0VgammaDt,
			   	CUstream* stream, int Npart){

	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		temperature_scaleKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (
						noise[dev], 
						alpha[dev], alphaMul,
						temp[dev], kB2tempMul,
						mSat[dev], msatMul,
						mu0VgammaDt, Npart);
	}
}

#ifdef __cplusplus
}
#endif
