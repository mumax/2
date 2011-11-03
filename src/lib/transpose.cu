/**
 * @author Arne Vansteenkiste
 */
#include "transpose.h"

#include "gpu_safe.h"
#include "gpu_conf.h"
#include "multigpu.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  float real;
  float imag;
}complex;

/// The size of matrix blocks to be loaded into shared memory.
#define BLOCKSIZE 16

// cross-device complex transpose-pad, aka. the dragon kernel.
__global__ void xdevTransposePadKernel(complex* output, complex* input, int N1, int N2, int N0, int chunkN2){

  	__shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

	for(int x=0; x<N0; x++){ // could take this out of kernel if we stream over planes...

   		 // index of the block inside the blockmatrix
   		 int BI = blockIdx.x;
   		 int BJ = blockIdx.y;

   		 // "minor" indices inside the tile
   		 int i = threadIdx.x;
   		 int j = threadIdx.y;
   		 
   		 // "major" indices inside the entire matrix
   		 int I = BI * BLOCKSIZE + i;
   		 int J = BJ * BLOCKSIZE + j;

   		 if((I < N1) && (J < chunkN2)){ // must be within this device's chunk (chunkN2)
   		   block[j][i] = input[x*N1*N2 + J * N1 + I]; // but indexes in total data (N2)
   		 }
   		 
   		 __syncthreads();

   		 
   		 // Major indices with transposed blocks but not transposed minor indices
   		 int It = BJ * BLOCKSIZE + i;
   		 int Jt = BI * BLOCKSIZE + j;

   		 if((It < chunkN2) && (Jt < N1)){
   		   output[x*N1*N2 + Jt * N2 + It] = block[i][j];
   		 }
    
   		 __syncthreads();

	}
}


void transposePadYZAsync(float** output_f, float** input_f, int N0, int N1Part, int N2, int N2Pad, CUstream* stream){
	int nDev = nDevice();

	N2 /= 2; // number of complexes
	complex** input = (complex**)input_f;
	complex** output = (complex**)output_f;

	// each chunk has a different destination device
	int chunkN2 = divUp(N2, nDev);
	int chunkN1 = N1Part;  // *N0;
	// chunkN0 = N0

	// divide each chunk in shmem blocks
    dim3 gridsize(divUp(chunkN2, BLOCKSIZE), divUp(chunkN1, BLOCKSIZE), 1);
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
	
	for (int dev = 0; dev < nDev; dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
	
		for(int chunk = 0; chunk < nDev; chunk++){
			// source device = dev
			// target device = chunk
			complex* src = &(input[dev][chunk*chunkN2]); // offset device pointer to start of chunk
			complex* dst = &(output[chunk][dev*chunkN2]); // offset device pointer to start of chunk

    		xdevTransposePadKernel<<<gridsize, blocksize, 0, stream[dev]>>>(dst, src, N2, N1Part, N0, chunkN2); // yes N1-N2 are reversed.
		
		}
	}
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void transposeComplexYZKernel(complex* output, complex* input, int N1, int N2, int N)
{
  __shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  for (int x=0; x<N; x++){
    // index of the block inside the blockmatrix
    int BI = blockIdx.x;
    int BJ = blockIdx.y;

    // "minor" indices inside the tile
    int i = threadIdx.x;
    int j = threadIdx.y;

    {
      // "major" indices inside the entire matrix
      int I = BI * BLOCKSIZE + i;
      int J = BJ * BLOCKSIZE + j;

      if((I < N1) && (J < N2)){
        block[j][i] = input[x*N1*N2 + J * N1 + I];
      }
    }
    __syncthreads();

    {
      // Major indices with transposed blocks but not transposed minor indices
      int It = BJ * BLOCKSIZE + i;
      int Jt = BI * BLOCKSIZE + j;

      if((It < N2) && (Jt < N1)){
        output[x*N1*N2 + Jt * N2 + It] = block[i][j];
      }
    }
    __syncthreads();
  }
  
  return;
}

void transposeComplexYZAsyncPart(float** output, float** input, int N0, int N1, int N2, CUstream* stream){
    N2 /= 2; // number of complex
    dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
    	transposeComplexYZKernel<<<gridsize, blocksize, 0, stream[dev]>>>((complex*)output[dev], (complex*)input[dev], N2, N1, N0);
	}
}




#ifdef __cplusplus
}
#endif

