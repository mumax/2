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
__global__ void currentDensityKern(float* jx, float* jy, float* jz, 
								   float* Ex, float* Ey, float* Ez,
								   float* EyPart0, float* EyPart2,
								   float* rmap, float rMul,
								   float* rPart0, float* rPart2,
								   int N0, int N1Part, int N2, 
								   int wrap0, int wrap2, 
								   //float cellx, float celly, float cellz,
								   int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1Part*N2 + j*N2 + k; // linear array index
  
  if (j < N1Part && k < N2){

	// central cell
    float r1 = rmap[I] * rMul;
	float E1 = Ex[I];

    // neighbors in X direction
	{
	float E0 = 0;
	float r0 = 1.0f/0.0f;
    if (i-1 >= 0){                                // neighbor in bounds...
      int idx = (i-1)*N1Part*N2 + j*N2 + k;       // ... no worries
	  E0 = Ex[idx];
	  r0 = rmap[idx] * rMul;
    } else {                                      // neighbor out of bounds...
		if(wrap0){                                // ... PBC?
			int idx = (N0-1)*N1Part*N2 + j*N2 + k;// yes: wrap around!
	  		E0 = Ex[idx];
	  		r0 = rmap[idx] * rMul;
		}
    }
	float j0 = (E1+E0) / (r1+r0);

	float E2 = 0;
	float r2 = 1.0f/0.0f;
 	if (i+1 < N0){
      int idx = (i+1)*N1Part*N2 + j*N2 + k;
	  E2 = Ex[idx];
	  r2 = rmap[idx] * rMul;
    } else {
		if(wrap0){
			int idx = (0)*N1Part*N2 + j*N2 + k;
	  		E2 = Ex[idx];
	  		r2 = rmap[idx] * rMul;
		}
    } 
	float j2 = (E1+E2) / (r1+r2);

	jx[I] = 0.5f*(j0+j2);
	}


    // neighbors in Z direction
	{
	float E0 = 0;
	float r0 = 1.0f/0.0f;
    if (k-1 >= 0){                                
      int idx = i*N1Part*N2 + j*N2 + (k-1);
	  E0 = Ez[idx];
	  r0 = rmap[idx] * rMul;
    } else {                                     
		if(wrap2){                              
  			int idx = i*N1Part*N2 + j*N2 + (N2-1);
	  		E0 = Ez[idx];
	  		r0 = rmap[idx] * rMul;
		}
    }
	float j0 = (E1+E0) / (r1+r0);

	float E2 = 0;
	float r2 = 1.0f/0.0f;
 	if (k+1 < N2){
  	  int idx =  i*N1Part*N2 + j*N2 + (k+1);
	  E2 = Ez[idx];
	  r2 = rmap[idx] * rMul;
    } else {
		if(wrap2){
  	        int idx = i*N1Part*N2 + j*N2 + (0);
	  		E2 = Ez[idx];
	  		r2 = rmap[idx] * rMul;
		}
    } 
	float j2 = (E1+E2) / (r1+r2);

	jz[I] = 0.5f*(j0+j2);
	}

    // Here be dragons.
    // neighbors in Y direction
	{
	float E0 = 0;
	float r0 = 1.0f/0.0f;
    if (j-1 >= 0){                                     // neighbor in bounds...
      int idx = i*N1Part*N2 + (j-1)*N2 + k;            // ...no worries
	  E0 = Ey[idx];
	  r0 = rmap[idx] * rMul;
    } else {                                           // neighbor out of bounds...
    	if(EyPart0 != NULL){                           // there is an adjacent part (either PBC or multi-GPU)
    		int idx = i*N1Part*N2 + (N1Part-1)*N2 + k; // take value from other part (either PBC or multi-GPU)
	  		E0 = Ey[idx];
	  		r0 = rmap[idx] * rMul;
    	}
    }
	float j0 = (E1+E0) / (r1+r0);

	float E2 = 0;
	float r2 = 1.0f/0.0f;
    if (j+1 < N1Part){
      int idx = i*N1Part*N2 + (j+1)*N2 + k;
	  E2 = Ey[idx];
	  r2 = rmap[idx] * rMul;
    } else {
    	if(EyPart2 != NULL){
    		int idx = i*N1Part*N2 + (0)*N2 + k;
	  		E2 = Ey[idx];
	  		r2 = rmap[idx] * rMul;
    	}
    } 
	float j2 = (E1+E2) / (r1+r2);

	jy[I] = 0.5f*(j0+j2);
	}
  }
}



#define BLOCKSIZE 16
void currentDensityAsync(float** jx, float** jy, float** jz, float** Ex, float** Ey, float** Ez, float** rMap, float rMul, 
						int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, 
						//float cellx, float celly, float cellz, 
						CUstream* streams){

  assert(rMap != NULL);

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
				jx[dev], jy[dev], jz[dev],
				Ex[dev], Ey[dev], Ez[dev],
				EyPart0, EyPart2, 
			    rMap[dev], rMul, rPart0, rPart2,
				N0, N1Part, N2, 
				periodic0, periodic2, 
				//cellx, celly, cellz,
				i);
		}
	}
}


#ifdef __cplusplus
}
#endif

