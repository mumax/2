#include "exchange6.h"
#include "current.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 2D, plane per plane, i=plane index
__global__ void currentDensityKern(float* jx, float* jy, float* jz, float* drho,
								   float* Ex, float* Ey, float* Ez,
								   float* EyPart0, float* EyPart2,
								   float* rmap, float rMul,
								   float* rPart0, float* rPart2,
								   int N0, int N1Part, int N2, 
								   int wrap0, int wrap2, 
								   float cellx, float celly, float cellz, int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1Part*N2 + j*N2 + k; // linear array index
  
  if (j < N1Part && k < N2){

//    float r;
//    if (rmap==NULL){
//      r = rMul;
//	}else{
//      r = rMul * rmap[I];
//	}

	
	float Ex0 = Ex[I];
    // neighbors in X direction
	int idx;
    if (i-1 >= 0){                                // neighbor in bounds...
      idx = (i-1)*N1Part*N2 + j*N2 + k;           // ... no worries
    } else {                                      // neighbor out of bounds...
		if(wrap0){                                // ... PBC?
			idx = (N0-1)*N1Part*N2 + j*N2 + k;    // yes: wrap around!
		}else{                                    
      		idx = I;                              // no: use central m (Neumann BC) 
		}
    }
	float Ex1 = Ex[idx];

	float j1 = (Ex0+Ex1) / (r0+r1)

 	if (i+1 < N0){
      idx = (i+1)*N1Part*N2 + j*N2 + k;
    } else {
		if(wrap0){
			idx = (0)*N1Part*N2 + j*N2 + k;
		}else{
      		idx = I;
		}
    } 
	float Ex2 = Ex[idx]; 

  //  float H = Aex2_Mu0Msat * cellx_2 * ((m1-m0) + (m2-m0));

  //  // neighbors in Z direction
  //  if (k-1 >= 0){
  //    idx = i*N1Part*N2 + j*N2 + (k-1);
  //  } else {
  //  	if(wrap2){
  //  		idx = i*N1Part*N2 + j*N2 + (N2-1);
  //  	}else{
  //    		idx = I;
  //  	}
  //  }
  //  m1 = m[idx];

  //  if (k+1 < N2){
  //    idx =  i*N1Part*N2 + j*N2 + (k+1);
  //  } else {
  //  	if(wrap2){
  //  		idx = i*N1Part*N2 + j*N2 + (0);
  //  	}else{
  //    		idx = I;
  //  	}
  //  } 
  //  m2 = m[idx];
  // 
  //  H += Aex2_Mu0Msat * cellz_2 * ((m1-m0) + (m2-m0));

  //  // Here be dragons.
  //  // neighbors in Y direction
  //  if (j-1 >= 0){                                 // neighbor in bounds...
  //    idx = i*N1Part*N2 + (j-1)*N2 + k;            // ...no worries
  //    m1 = m[idx];
  //  } else {                                       // neighbor out of bounds...
  //  	if(mPart0 != NULL){                        // there is an adjacent part (either PBC or multi-GPU)
  //  		idx = i*N1Part*N2 + (N1Part-1)*N2 + k; // take value from other part (either PBC or multi-GPU)
  //  		m1 = mPart0[idx];
  //  	}else{                                     // no adjacent part: use central m (Neumann BC)
  //    		m1 = m[I];
  //  	}
  //  }

  //  if (j+1 < N1Part){
  //    idx = i*N1Part*N2 + (j+1)*N2 + k;
  //    m2 = m[idx];
  //  } else {
  //  	if(mPart2 != NULL){
  //  		idx = i*N1Part*N2 + (0)*N2 + k;
  //          m2 = mPart2[idx];
  //  	}else{
  //  		m2 = m[I];
  //  	}
  //  } 
  //  H += Aex2_Mu0Msat * celly_2 * ((m1-m0) + (m2-m0));

  //  // Write back to global memory
  //  h[I] = H;

  }
}



#define BLOCKSIZE 16
void currentDensityAsync(float** jx, float** jy, float** jz, float** drho, float** Ex, float** Ey, float** Ez, float** rMap, float rMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellx, float celly, float cellz, CUstream* streams){
  dim3 gridsize(divUp(N1Part, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  //int NPart = N0 * N1Part * N2;

	int nDev = nDevice();

	for (int dev = 0; dev < nDev; dev++) {

		gpu_safe(cudaSetDevice(deviceId(dev)));

		// set up adjacent parts
		float* EyPart0 = Ey[mod(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
		float* EyPart2 = Ey[mod(dev+1, nDev)];  // parts wrap around...
		float* rPart0 = rMap[mod(dev-1, nDev)];
		float* rPart2 = rMap[mod(dev+1, nDev)];
		if(periodic1 == 0){                     // unless there are no PBCs...
			if(dev == 0){
				EyPart0 = NULL;
				rPart0 = NULL;
			}
			if(dev == nDev-1){
				EyPart2 = NULL;
				rPart2 = NULL;
			}
		}

		for(int i=0; i<N0; i++){   // for all layers. TODO: 2D version
			currentDensityKern<<<gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(
				jx[dev], jy[dev], jz[dev], drho[dev],
				Ex[dev], Ey[dev], Ez[dev],
				EyPart0, EyPart2, 
			    rMap[dev], rMul, rPart0, rPart2,
				N0, N1Part, N2, 
				periodic0, periodic2, 
				cellx, celly, cellz, i);
		}
	}
}


#ifdef __cplusplus
}
#endif

