
#include "zhang_li_torque.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <cuda.h>


#ifdef __cplusplus
extern "C" {
#endif




///@todo Not correct at the edges with a normmap!


///@internal
__global__ void _gpu_spintorque_deltaM(float* mx, float* my, float* mz,
                                       float* hx, float* hy, float* hz,
                                       float* alpha, float* bj, float* cj,
                                       float* Msat,
                                       float ux, float uy, float uz,
                                       float* jmapx, float* jmapy, float* jmapz,
                                       float dt_gilb,
                                       int Npart,
                                       int dev,
                                       int Nx, int NyPart, int Nz){
	int i = threadindex;
	if (i < Npart && Msat[i]!=0.0f) {

		int x = i%Nx;
		int y = ((i/Nx)%NyPart+(NyPart*dev));
		int z = i/(Nx*NyPart);

		// space-dependent map
		if(jmapx != NULL){
			ux *= jmapx[i];
		}
		if(jmapy != NULL){
			uy *= jmapy[i];
		}
		if(jmapz != NULL){
			uz *= jmapz[i];
		}


		// (1) calculate the directional derivative of (mx, my mz) along (ux,uy,uz).
		// Result is (diffmx, diffmy, diffmz) (= Hspin)
		// (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)

		//float m0x = mx[i*N1*N2 + j*N2 + k];
		float mx1 = 0.f, mx2 = 0.f, my1 = 0.f, my2 = 0.f, mz1 = 0.f, mz2 = 0.f;

		// derivative in X direction
		if (x-1 >= 0 && Msat[i-1] != 0.0f){
			int idx = i-1;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		} else {
		  // How to handle edge cells?
		  // * leaving the m value zero gives too big a gradient
		  // * setting it to the central value gives the actual gradient / 2, should not hurt
		  // * problem with nonuniform norm!! what if a neighbor has zero norm (but still lies in the box)?
		  int idx = i;
		  mx1 = mx[idx];
		  my1 = my[idx];
		  mz1 = mz[idx];
		}
		if (x+1 < Nx && Msat[i+1] != 0.0f){
			int idx = i+1;
			mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
		} else {
			int idx = i;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		}
		float diffmx = ux * (mx2 - mx1);
		float diffmy = ux * (my2 - my1);
		float diffmz = ux * (mz2 - mz1);


		// derivative in Y direction
		if (y-1 >= 0 && Msat[i-NyPart] != 0.0f){
			int idx = i - NyPart;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		} else {
			int idx = i;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		}
		if (y+1 < Npart/(Nx*Nz) && Msat[i+NyPart] != 0.0f){
			int idx = i+NyPart;
			mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
		} else {
			int idx = i;
			mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
		}
		diffmx += uy * (mx2 - mx1);
		diffmy += uy * (my2 - my1);
		diffmz += uy * (mz2 - mz1);


		// derivative in Z direction
		if (z-1 >= 0 && Msat[i-Nx*NyPart] != 0.0f){
			int idx = i - Nx*NyPart;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		} else {
			int idx = i;
			mx1 = mx[idx];
			my1 = my[idx];
			mz1 = mz[idx];
		}
		if (z+1 < Nz && Msat[i+Nx*NyPart] != 0.0f){
			int idx = i + Nx*NyPart;
			mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
		} else {
			int idx = i;
			mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
		}
		diffmx += uz * (mx2 - mx1);
		diffmy += uz * (my2 - my1);
		diffmz += uz * (mz2 - mz1);


		//(2) torque terms

		// H
		float Hx = hx[i];
		float Hy = hy[i];
		float Hz = hz[i];
		float Cj	 	= cj[i];
		float Bj 		= bj[i];
		float Alpha 	= alpha[i];
		float Ms		= Msat[i];

		// m
		float Mx = mx[i], My = my[i], Mz = mz[i];

		// Hp (Hprecess) = H + epsillon Hspin
		float Hpx = Hx + Cj * diffmx / Ms;
		float Hpy = Hy + Cj * diffmy / Ms;
		float Hpz = Hz + Cj * diffmz / Ms;

		// - m cross Hprecess
		float _mxHpx = -My * Hpz + Hpy * Mz;
		float _mxHpy =  Mx * Hpz - Hpx * Mz;
		float _mxHpz = -Mx * Hpy + Hpx * My;

		// Hd Hdamp = alpha*H + beta*Hspin
		float Hdx = Alpha * Hx + Bj * diffmx / Ms;
		float Hdy = Alpha * Hy + Bj * diffmy / Ms;
		float Hdz = Alpha * Hz + Bj * diffmz / Ms;

		// - m cross Hdamp
		float _mxHdx = -My * Hdz + Hdy * Mz;
		float _mxHdy =  Mx * Hdz - Hdx * Mz;
		float _mxHdz = -Mx * Hdy + Hdx * My;


		// - m cross (m cross Hd)
		float _mxmxHdx =  My * _mxHdz - _mxHdy * Mz;
		float _mxmxHdy = -Mx * _mxHdz + _mxHdx * Mz;
		float _mxmxHdz =  Mx * _mxHdy - _mxHdx * My;

		hx[i] = dt_gilb * (-_mxHpx + _mxmxHdx);
		hy[i] = dt_gilb * (-_mxHpy + _mxmxHdy);
		hz[i] = dt_gilb * (-_mxHpz + _mxmxHdz);
	}

}



void gpu_spintorque_deltaM(float** mx, float** my, float** mz,
						   float** hx, float** hy, float** hz,
						   float** alpha,
						   float** bj,
						   float** cj,
						   float** Msat,
						   float ux, float uy, float uz,
						   float** jx, float** jy, float** jz,
						   float dt_gilb,
						   CUstream* stream,
						   int Npart,
						   int Nx, int NyPart, int Nz)
{

	dim3 gridSize, blockSize;
	make1dconf(Npart, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		assert(mx[dev] != NULL);
		assert(my[dev] != NULL);
		assert(mz[dev] != NULL);
		assert(hx[dev] != NULL);
		assert(hy[dev] != NULL);
		assert(hz[dev] != NULL);
		assert(jx[dev] != NULL);
		assert(jy[dev] != NULL);
		assert(hz[dev] != NULL);
		assert(alpha[dev] != NULL);
		assert(bj[dev] != NULL);
		assert(cj[dev] != NULL);
		assert(Msat[dev] != NULL);
		gpu_safe(cudaSetDevice(deviceId(dev)));
		_gpu_spintorque_deltaM <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (mx[dev],my[dev],mz[dev],
																						hx[dev],hy[dev],hz[dev],
																						alpha[dev],
																						bj[dev],
																						cj[dev],
																						Msat[dev],
																						ux,uy,uz,
																						jx[dev],jy[dev],jz[dev],
																						dt_gilb,
																						Npart,
																						dev,
																						Nx, NyPart,Nz);
	}
}




/*
#define BLOCKSIZE 16

///@todo this is a slowish test implementation, use shared memory to avoid multiple reads (at least per plane)
__global__ void _gpu_directional_diff2D(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2, int i){

//int i = i;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < N1 && k < N2){

    float m0 = in[i*N1*N2 + j*N2 + k];
    float mx1, mx2;

    if (i-1 >= 0){ mx1 = in[(i-1)*N1*N2 + j*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (i+1 < N0){ mx2 = in[(i+1)*N1*N2 + j*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    float answer =  ux * (mx2 - mx1);

    if (j-1 >= 0){ mx1 = in[(i)*N1*N2 + (j-1)*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (j+1 < N1){ mx2 = in[(i)*N1*N2 + (j+1)*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    answer += uy * (mx2 - mx1);

    if (k-1 >= 0){ mx1 = in[(i)*N1*N2 + (j)*N2 + (k-1)]; } else { mx1 = m0; } // not 100% accurate
    if (k+1 < N2){ mx2 = in[(i)*N1*N2 + (j)*N2 + (k+1)]; } else { mx2 = m0; } // not 100% accurate
    answer += uz * (mx2 - mx1);

    out[i*N1*N2 + j*N2 + k] = answer;
  }
}


void gpu_directional_diff2D_async(float ux, float uy, float uz, float *input, float *output, int N0, int N1, int N2, int i){
    dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_directional_diff2D<<<gridsize, blocksize>>>(ux, uy, uz, input, output, N0, N1, N2, i);
}

void gpu_directionial_diff(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2){
  for(int i=0; i<N0; i++){
    gpu_directional_diff2D_async(ux, uy, uz, &in[i*N1*N2], &out[i*N1*N2], N0, N1, N2, i);
  }
  gpu_sync();
}


*/
#ifdef __cplusplus
}
#endif


