#include "copypad.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @author Arne Vansteenkiste & Ben Van de Wiele

/// @internal Copies a matrix ("block") into dst, a larger matrix
/// The position of of the block in dst is block*S2 along the N2 direction.
__global__ void insertBlockZKern(float* dst, int D2, float* src, int S0, int S1, int S2, int block){
  
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int k = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<S0; i++){
      if(j<S1 && k<S2){ // we are in the source array
        dst[i*S1*D2 + j*D2 + block*S2 + k] = src[i*S1*S2 + j*S2 + k];
      }
    }
}


void insertBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1); // range over source size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    insertBlockZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst[dev], D2, src[dev], S0, S1Part, S2, block);///@todo stream or loop in kernel
  }
}


/// @internal Extracts a matrix ("block") from src, a larger matrix
/// The position of of the block in src is block*D2 along the N2 direction.
__global__ void extractBlockZKern(float* dst, int D0, int D1, int D2, float *src, int S2, int block){
  
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  
  for(int i=0; i<D0; i++){
    if(j<D1 && k<D2){ // we are in the destination array
      dst[i*D1*D2 + j*D2 + k] = src[i*D1*S2 + j*S2 + block*D2 + k];
    }
  }
}


void extractBlockZAsync(float **dst, int D0, int D1Part, int D2, float **src, int S2, int block, CUstream *streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1Part, BLOCKSIZE), 1); // range over destination size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    extractBlockZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst[dev], D0, D1Part, D2, src[dev], S2, block);///@todo stream or loop in kernel
  }
}



__global__ void zeroArrayKern(float *A, int N){
  
  int i = threadindex;
  
  if (i<N){
    A[i] = 0.0f;
  }
}

void zeroArrayAsync(float **A, int length, CUstream *streams){

  dim3 gridSize, blockSize;
  make1dconf(length, &gridSize, &blockSize);
  
  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    zeroArrayKern<<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>>( A[dev], length );
  }
}



/// @internal Does padding and unpadding of a 3D matrix.  Padding in the y-direction is only correct when 1 GPU is used!!
/// Fills padding space with zeros.
__global__ void copyPad3DKern(float* dst, int D0, int D1, int D2, float* src, int S0, int S1, int S2){
  
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int k = blockIdx.x * blockDim.x + threadIdx.x;

  // this check makes it work for padding as well as for unpadding.
  // 2 separate functions are probably not more efficient
  // due to memory bandwidth limitations
  for (int i=0; i<D0; i++){
    if (j<D1 && k<D2){ // if we are in the destination array we should write something
      if(i<S0 && j<S1 && k<S2){ // we are in the source array: copy the source
        dst[i*D1*D2 + j*D2 + k] = src[i*S1*S2 + j*S2 + k];
      }else{ // we are out of the source array: write zero
        dst[i*D1*D2 + j*D2 + k] = 0.0f; 
      }
    }
  }
}

void copyPad3DAsync(float** dst, int D0, int D1, int D2, float** src, int S0, int S1, int S2, int Ncomp, CUstream* streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1); // range over destination size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    for (int i=0; i<Ncomp; i++){
      float* src3D = &(src[dev][i*S0*S1*S2]);
      float* dst3D = &(dst[dev][i*D0*D1*D2]); //D1==S1
      copyPad3DKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst3D, D0, D1, D2, src3D, S0, S1, S2);
    }
  }
}




/// @internal Does padding and unpadding ONLY Z-DIRECTION
///	Fills padding space with zeros.
__global__ void copyPadZKern(float* dst, int D2, float* src, int S0, int S1, int S2){
  
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   int k = blockIdx.x * blockDim.x + threadIdx.x;
   int D1 = S1; ///@todo:rm
   int D0 = S0;

	// this check makes it work for padding as well as for unpadding.
	// 2 separate functions are probably not more efficient
	// due to memory bandwidth limitations
  for (int i=0; i<D0; i++){
    if (j<D1 && k<D2){ // if we are in the destination array we should write something
      if(j<S1 && k<S2){ // we are in the source array: copy the source
        dst[i*D1*D2 + j*D2 + k] = src[i*S1*S2 + j*S2 + k];
      }else{ // we are out of the source array: write zero
        dst[i*D1*D2 + j*D2 + k] = 0.0f; 
      }
    }
  }
}

void copyPadZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, CUstream* streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1); // range over destination size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
    copyPadZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst[dev], D2, src[dev], S0, S1Part, S2);///@todo stream or loop in kernel
	}
}


#ifdef __cplusplus
}
#endif

