#include "exchange6.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

/// 2D, plane per plane, i=plane index
__global__ void exchange6Kern(float* h, float* m, float* mSat_map, float* Aex_map, float Aex2_Mu0Msat_mul, float* mPart0, float* mPart2,
                               int N0, int N1Part, int N2,
                               int wrap0, int wrap2,
                               float cellx_2, float celly_2, float cellz_2, 
                               int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1Part*N2 + j*N2 + k; // linear array index
  
  if (j < N1Part && k < N2){

	float mSat_mask;
	if (mSat_map ==NULL){
		mSat_mask = 1.0f;
	}else{
		mSat_mask = mSat_map[I];
		if (mSat_mask == 0.0f){
			mSat_mask = 1.0f; // do not divide by zero
		}
	}

    float Aex2_Mu0Msat; // 2 * Aex / Mu0 * Msat
    if (Aex_map==NULL){
      Aex2_Mu0Msat = Aex2_Mu0Msat_mul / mSat_mask;
	}else{
      Aex2_Mu0Msat = (Aex2_Mu0Msat_mul / mSat_mask) * Aex_map[I];
	}



	float m0 = m[I]; // mag component of central cell 
	float m1, m2 ;   // mag component of neighbors in 2 directions
	
    // neighbors in X direction
	int idx;
    if (i-1 >= 0){                                // neighbor in bounds...
      idx = (i-1)*N1Part*N2 + j*N2 + k;           // ... no worries
    } else {                                      // neighbor out of bounds...
		if(wrap0){                                // ... PBC?
			idx = (N0-1)*N1Part*N2 + j*N2 + k;    // yes: wrap around!
		}else{                                    
		  //idx = I;                              // no: use central m (Neumann BC)
		  idx = -1;				  // no: use zero-value outside, because it is important for LLBr 
		}
    }
	m1 = (idx < 0) ? 0.0f : m[idx];

 	if (i+1 < N0){
      idx = (i+1)*N1Part*N2 + j*N2 + k;
    } else {
		if(wrap0){
			idx = (0)*N1Part*N2 + j*N2 + k;
		}else{
      		//idx = I;
		idx = -1;				  // no: use zero-value outside, because it is important for LLBr 
		}
    } 
	m2 = (idx < 0) ? 0.0f : m[idx]; 

    float H = Aex2_Mu0Msat * cellx_2 * ((m1-m0) + (m2-m0));

    // neighbors in Z direction
    if (k-1 >= 0){
      idx = i*N1Part*N2 + j*N2 + (k-1);
    } else {
		if(wrap2){
			idx = i*N1Part*N2 + j*N2 + (N2-1);
		}else{
      		//idx = I;
		idx = -1;				  // no: use zero-value outside, because it is important for LLBr 
		}
    }
	m1 = (idx < 0) ? 0.0f : m[idx];

 	if (k+1 < N2){
      idx =  i*N1Part*N2 + j*N2 + (k+1);
    } else {
		if(wrap2){
			idx = i*N1Part*N2 + j*N2 + (0);
		}else{
      		//idx = I;
		idx = -1;				  // no: use zero-value outside, because it is important for LLBr
		}
    } 
	m2 = (idx < 0) ? 0.0f : m[idx];
   
    H += Aex2_Mu0Msat * cellz_2 * ((m1-m0) + (m2-m0));

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
	  		//m1 = m[I];
			m1 = 0.0f;
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
			//m2 = m[I];
			m2 = 0.0f;
		}
    } 
    H += Aex2_Mu0Msat * celly_2 * ((m1-m0) + (m2-m0));

	// Write back to global memory
    h[I] = H;
  }
  
}


#define BLOCKSIZE 16
__export__ void exchange6Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float** msat, float** aex, float Aex2_mu0MsatMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams){

  dim3 gridsize(divUp(N1Part, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  //int NPart = N0 * N1Part * N2;

  float cellx_2 = 1/(cellSizeX * cellSizeX);
  float celly_2 = 1/(cellSizeY * cellSizeY);
  float cellz_2 = 1/(cellSizeZ * cellSizeZ);
  //printf("exchange factors %g %g %g\n", fac0, fac1, fac2); // OK

	int nDev = nDevice();

	float** H = hx;
    float** M = mx;
	for(int c=0; c<3; c++){        // for all 3 components
		if (c==0){H = hx; M = mx;}
		if (c==1){H = hy; M = my;}
		if (c==2){H = hz; M = mz;}

		for (int dev = 0; dev < nDev; dev++) {
	
			gpu_safe(cudaSetDevice(deviceId(dev)));
	
			// set up adjacent parts
			float* mPart0 = M[Mod(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
			float* mPart2 = M[Mod(dev+1, nDev)];  // parts wrap around...
			if(periodic1 == 0){                     // unless there are no PBCs...
				if(dev == 0){
					mPart0 = NULL;
				}
				if(dev == nDev-1){
					mPart2 = NULL;
				}
			}
			//printf("exch dev=%d mPart0=%p mPart2=%p\n", dev, mPart0, mPart2); // OK

  			for(int i=0; i<N0; i++){   // for all layers. TODO: 2D version
    			exchange6Kern<<<gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(H[dev], M[dev], msat[dev], aex[dev], Aex2_mu0MsatMul, mPart0, mPart2, N0, N1Part, N2, periodic0, periodic2, cellx_2, celly_2, cellz_2, i);
			}
  		}
	}
}


#ifdef __cplusplus
}
#endif

