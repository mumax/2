
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


///@internal
__global__ void temperature_scaleAnisKern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

			    float* __restrict__ mu_xx,
				float* __restrict__ mu_yy,
				float* __restrict__ mu_zz,
				float* __restrict__ mu_yz,
				float* __restrict__ mu_xz,
				float* __restrict__ mu_xy,
				
				float* __restrict__ tempMask,
				float* __restrict__ msatMask, 
				float* __restrict__ msat0T0Mask,
				
				const float muMul_xx,
				const float muMul_yy,
				const float muMul_zz,
				const float muMul_yz,
				const float muMul_xz,
				const float muMul_xy,
				
				const float KB2tempMul,
				const float mu0VgammaDtMSatMul,
				
			   	int Npart){


	int i = threadindex;
	
	if (i < Npart) {
		
		float msat0T0 = (msat0T0Mask != NULL) ? msat0T0Mask[i] : 1.0;
		if (msat0T0 == 0.0f) {
			 hx[i] = 0.0f;
			 hy[i] = 0.0f;
			 hz[i] = 0.0f;
			 return;
		}  
		
		float3 H = make_float3(hx[i], hy[i], hz[i]);
			
		float3 mu_H;
		
		float m_xx = (mu_xx != NULL) ? mu_xx[i] * muMul_xx : muMul_xx;
		float m_xy = (mu_xy != NULL) ? mu_xy[i] * muMul_xy : muMul_xy;
		float m_xz = (mu_xz != NULL) ? mu_xz[i] * muMul_xz : muMul_xz;
		m_xx = sqrtf(m_xx);
		m_xy = sqrtf(m_xy);
		m_xz = sqrtf(m_xz);
		
		mu_H.x = m_xx * H.x + m_xy * H.y + m_xz * H.z;
		
		float m_yy = (mu_yy != NULL) ? mu_yy[i] * muMul_yy : muMul_yy;
		float m_yz = (mu_yz != NULL) ? mu_yz[i] * muMul_yz : muMul_yz;
		m_yy = sqrtf(m_yy);
		m_yz = sqrtf(m_yz);
		
	    mu_H.y = m_xy * H.x + m_yy * H.y + m_yz * H.z;
		
		float m_zz = (mu_zz != NULL) ? mu_zz[i] * muMul_zz : muMul_zz;
		m_zz = sqrtf(m_zz);
		
        mu_H.z = m_xz * H.x + m_yz * H.y + m_zz * H.z;

		float msat = (msatMask != NULL) ? msatMask[i] : 1.0;
		float T = (tempMask != NULL) ? tempMask[i] : 1.0;
		float pre = sqrtf((T * KB2tempMul)/(mu0VgammaDtMSatMul * msat));
		hx[i] = pre * mu_H.x;
		hy[i] = pre * mu_H.y;
		hz[i] = pre * mu_H.z;
		
	}
}


__export__ void temperature_scaleAnizNoise(float** hx, float** hy, float** hz,
			   	float** mu_xx, 
			   	float** mu_yy, 
			   	float** mu_zz, 
			   	float** mu_yz, 
			   	float** mu_xz, 
			   	float** mu_xy, 
			   	float** tempMask, 
			   	float** msatMask,
			   	float** msat0T0Mask,
			   	
			   	float muMul_xx,
				float muMul_yy,
				float muMul_zz,
				float muMul_yz,
				float muMul_xz,
				float muMul_xy,
				
			    float KB2tempMul, 
			   	float mu0VgammaDtMSatMul,
			   	CUstream* stream, 
			   	int Npart){

	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		temperature_scaleAnisKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (
						hx[dev], hy[dev], hz[dev],
						mu_xx[dev],
						mu_yy[dev],
						mu_zz[dev],
						mu_yz[dev],
						mu_xz[dev],
						mu_xy[dev],
						tempMask[dev], 
						msatMask[dev], 
						msat0T0Mask[dev],
						muMul_xx,
						muMul_yy,
						muMul_zz,
						muMul_yz,
						muMul_xz,
						muMul_xy,
						KB2tempMul, 
						mu0VgammaDtMSatMul, Npart);
	}
}

#ifdef __cplusplus
}
#endif
