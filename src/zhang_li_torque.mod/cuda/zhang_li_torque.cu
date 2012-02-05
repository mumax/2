
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
__global__ void spintorque_deltaMKern(float* mx, float* my, float* mz,
									  float* mxPart0, float* myPart0, float* mzPart0,
									  float* mxPart2, float* myPart2, float* mzPart2,
                                      float* hx, float* hy, float* hz,
                                      float* alpha, float* bj, float* cj,
                                      float* Msat,
                                      float ux, float uy, float uz,
                                      float* jmapx, float* jmapy, float* jmapz,
                                      float dt_gilb,
                                      int N0, int N1Part, int N2,
                                      int i){
	//  i is passed
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	int I = i*N1Part*N2 + j*N2 + k; // linear array index

	if (j < N1Part && k < N2 && Msat[I]!=0.0f){

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

	    // neighbors in X direction
		int idx = (i-1)*N1Part*N2 + j*N2 + k;		// if neighbour in bounds and neighbour is magnetic keep it
	    if (i-1 < 0 || Msat[idx]==0.0f){			// else, neighbor out of bounds or non magnetic...
//			if(wrap0){								// ... PBC?
//				idx = (N0-1)*N1Part*N2 + j*N2 + k;	// yes: wrap around!
//			}else{
	      		idx = I;							// no: use central m (Neumann BC)
//			}
	    }
	    mx1 = mx[idx];
		my1 = my[idx];
		mz1 = mz[idx];

		idx = (i+1)*N1Part*N2 + j*N2 + k;
	 	if (i+1 >= N0 || Msat[idx]==0.0f){
//			if(wrap0){
//				idx = (0)*N1Part*N2 + j*N2 + k;
//			}else{
	      		idx = I;
//			}
	    }
	    mx2 = mx[idx];
		my2 = my[idx];
		mz2 = mz[idx];

		float diffmx = ux * (mx2 - mx1);
		float diffmy = ux * (my2 - my1);
		float diffmz = ux * (mz2 - mz1);

		idx = i*N1Part*N2 + j*N2 + (k-1);
	    // neighbors in Z direction
	    if (k-1 < 0 || Msat[idx]==0.0f){
//			if(wrap2){
//				idx = i*N1Part*N2 + j*N2 + (N2-1);
//			}else{
	      		idx = I;
//			}
	    }
	    mx1 = mx[idx];
		my1 = my[idx];
		mz1 = mz[idx];

		idx =  i*N1Part*N2 + j*N2 + (k+1);
	 	if (k+1 >= N2 || Msat[idx]==0.0f){
//			if(wrap2){
//				idx = i*N1Part*N2 + j*N2 + (0);
//			}else{
	      		idx = I;
//			}
	    }
	    mx2 = mx[idx];
		my2 = my[idx];
		mz2 = mz[idx];

		diffmx += uz * (mx2 - mx1);
		diffmy += uz * (my2 - my1);
		diffmz += uz * (mz2 - mz1);

		// Here be dragons.
	    // neighbors in Y direction
		idx = i*N1Part*N2 + (j-1)*N2 + k;
	    if (j-1 >= 0 && Msat[idx]!=0.0f){				// neighbour in bounds and neighbour is magnetic keep it
			mx1 = mx[idx];								// no worry
			my1 = my[idx];
			mz1 = mz[idx];
	    } else if (j-1 >= 0 && Msat[idx]==0.0f){		// neighbour in bounds but neighbour not magnetic
		    mx1 = mx[I];								// no adjacent magnetic part: use central m (Neumann BC)
			my1 = my[I];
			mz1 = mz[I];
	    } else {										// neighbor out of bounds...
			if(mxPart0 != NULL && myPart0 != NULL && mzPart0 != NULL){                        // there is an adjacent part (either PBC or multi-GPU)
				idx = i*N1Part*N2 + (N1Part-1)*N2 + k; // take value from other part (either PBC or multi-GPU)
				mx1 = mxPart0[idx];
				my1 = myPart0[idx];
				mz1 = mzPart0[idx];
			}else{                                     // no adjacent part: use central m (Neumann BC)
			    mx1 = mx[I];
				my1 = my[I];
				mz1 = mz[I];
			}
	    }

	    idx = i*N1Part*N2 + (j+1)*N2 + k;
	 	if (j+1 < N1Part && Msat[idx]!=0.0f){
		    mx2 = mx[idx];
			my2 = my[idx];
			mz2 = mz[idx];
	    } else if (j+1 < N1Part && Msat[idx]==0.0f){
		    mx2 = mx[I];
			my2 = my[I];
			mz2 = mz[I];
	    } else {
			if(mxPart2 != NULL && myPart2 != NULL && mzPart2 != NULL){
				idx = i*N1Part*N2 + (0)*N2 + k;
				mx2 = mxPart2[idx];
				my2 = myPart2[idx];
				mz2 = mzPart2[idx];
			}else{
			    mx2 = mx[I];
				my2 = my[I];
				mz2 = mz[I];
			}
	    }
		diffmx += uy * (mx2 - mx1);
		diffmy += uy * (my2 - my1);
		diffmz += uy * (mz2 - mz1);


		//(2) torque terms

		// H
		float Hx 	= hx[I];
		float Hy 	= hy[I];
		float Hz 	= hz[I];
		float Cj	= cj[I];
		float Bj 	= bj[I];
		float Alpha = alpha[I];
		float Ms	= Msat[I];

		// m
		float Mx = mx[I], My = my[I], Mz = mz[I];

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

		hx[I] = dt_gilb * (-_mxHpx + _mxmxHdx);
		hy[I] = dt_gilb * (-_mxHpy + _mxmxHdy);
		hz[I] = dt_gilb * (-_mxHpz + _mxmxHdz);
	}

}



int modz(int a, int b){
	return (a%b+b)%b;
}

#define BLOCKSIZEZ 16

void spintorque_deltaMAsync(float** mx, float** my, float** mz,
						   float** hx, float** hy, float** hz,
						   float** alpha,
						   float** bj,
						   float** cj,
						   float** Msat,
						   float ux, float uy, float uz,
						   float** jx, float** jy, float** jz,
						   float dt_gilb,
						   CUstream* stream,
						   int N0, int N1Part, int N2)
{

	dim3 gridSize(divUp(N1Part, BLOCKSIZEZ), divUp(N2, BLOCKSIZEZ));
	dim3 blockSize(BLOCKSIZEZ, BLOCKSIZEZ, 1);
	//int NPart = N0 * N1Part * N2;

	int nDev = nDevice();
	for (int dev = 0; dev < nDev; dev++) {

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

		// set up adjacent parts
		float* mxPart0 = mx[modz(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
		float* myPart0 = my[modz(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
		float* mzPart0 = mz[modz(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
		float* mxPart2 = mx[modz(dev+1, nDev)];  // parts wrap around...
		float* myPart2 = my[modz(dev+1, nDev)];  // parts wrap around...
		float* mzPart2 = mz[modz(dev+1, nDev)];  // parts wrap around...
// TODO: add periodic boundary conditions
//  		if(periodic1 == 0){                     // unless there are no PBCs...
		if(dev == 0){
			mxPart0 = NULL;
			myPart0 = NULL;
			mzPart0 = NULL;
		}
		if(dev == nDev-1){
			mxPart2 = NULL;
			myPart2 = NULL;
			mzPart2 = NULL;
		}
//		}
		for(int i=0; i<N0; i++){   // for all layers. TODO: 2D version
			spintorque_deltaMKern <<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>> (mx[dev], my[dev], mz[dev],
																						   mxPart0, myPart0, mzPart0,
																						   mxPart2, myPart2, mzPart2,
																						   hx[dev], hy[dev], hz[dev],
																						   alpha[dev], bj[dev], cj[dev],
																						   Msat[dev],
																						   ux, uy, uz,
																						   jx[dev], jy[dev], jz[dev],
																						   dt_gilb,
																						   N0, N1Part,N2,
																						   i);
		}
	}
//--------------------------------------------------------------------------------------------------
/*
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
	*/
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


