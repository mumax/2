#include "exchange6.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 2D, plane per plane, i=plane index
__global__ void exchange6Kern(float* h, float* m, float* mPart0, float* mPart2,
                               int N0, int N1Part, int N2,
                               int wrap0, int wrap2,
                               float fac0, float fac1, float fac2, 
                               int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1Part*N2 + j*N2 + k; // linear array index
  
  if (j < N1Part && k < N2){

	float m0 = m[I]; // mag component of central cell 
	float m1, m2 ;   // mag component of neighbors in 2 directions
	
	// Now add Neighbors.

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
	m1 = m[idx];

 	if (i+1 < N0){
      idx = (i+1)*N1Part*N2 + j*N2 + k;
    } else {
		if(wrap0){
			idx = (0)*N1Part*N2 + j*N2 + k;
		}else{
      		idx = I;
		}
    } 
	m2 = m[idx]; 

    float H = fac0 * ((m1-m0) + (m2-m0));

    // neighbors in Z direction
    if (k-1 >= 0){
      idx = i*N1Part*N2 + j*N2 + (k-1);
    } else {
		if(wrap2){
			idx = i*N1Part*N2 + j*N2 + (N2-1);
		}else{
      		idx = I;
		}
    }
	m1 = m[idx];

 	if (k+1 < N2){
      idx =  i*N1Part*N2 + j*N2 + (k+1);
    } else {
		if(wrap2){
			idx = i*N1Part*N2 + j*N2 + (0);
		}else{
      		idx = I;
		}
    } 
	m2 = m[idx];
   
    H += fac2 * ((m1-m0) + (m2-m0));

	// Here be dragons.
    // neighbors in Y direction
    if (j-1 >= 0){                                 // neighbor in bounds...
      idx = i*N1Part*N2 + (j-1)*N2 + k;            // ...no worries
	  m1 = m[idx];
    } else {                                       // neighbor out of bounds...
		if(mPart0 != NULL){                        // there is an adjacent part (either PBC or multi-GPU)
			idx = i*N1Part*N2 + (N1Part-1)*N2 + k; // take value from other part (either PBC or multi-GPU)
			m1 = mPart0[idx];
		}else{                                     // no adjacent part: use central m (Neumann BC)
	  		m1 = m[I];
		}
    }

 	if (j+1 < N1Part){
      idx = i*N1Part*N2 + (j+1)*N2 + k;
      m2 = m[idx];
    } else {
		if(mPart2 != NULL){
			idx = i*N1Part*N2 + (0)*N2 + k;
            m2 = mPart2[idx];
		}else{
			m2 = m[I];
		}
    } 
    H += fac1 * ((m1-m0) + (m2-m0));

	// Write back to global memory
    h[I] = H;
  }
  
}


// Python-style modulo (returns positive int)
int mod(int a, int b){
	return (a%b+b)%b;
}

#define BLOCKSIZE 16
void exchange6Async(float*** h, float*** m, float Aex, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams){

  dim3 gridsize(divUp(N1Part, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  //int NPart = N0 * N1Part * N2;

  float fac0 = Aex/(cellSizeX * cellSizeX);
  float fac1 = Aex/(cellSizeY * cellSizeY);
  float fac2 = Aex/(cellSizeZ * cellSizeZ);

	int nDev = nDevice();

	for(int c=0; c<3; c++){        // for all 3 components
	    float** H = h[c];
	    float** M = m[c];

		for (int dev = 0; dev < nDev; dev++) {
	
			gpu_safe(cudaSetDevice(deviceId(dev)));
	
			// set up adjacent parts
			float* mPart0 = M[mod(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
			float* mPart2 = M[mod(dev+1, nDev)];  // parts wrap around...
			if(periodic1 == 0){                     // unless there are no PBCs...
				if(dev == 0){
					mPart0 = NULL;
				}
				if(dev == nDev-1){
					mPart2 = NULL;
				}
			}

  			for(int i=0; i<N0; i++){   // for all layers. TODO: 2D version
    			exchange6Kern<<<gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(H[dev], M[dev], mPart0, mPart2, N0, N1Part, N2, periodic0, periodic2, fac0, fac1, fac2, i);
			}
  		}
	}
}


#ifdef __cplusplus
}
#endif

