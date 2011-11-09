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


/// @internal Single-GPU complex matrix transpose
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

