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
__global__ void insertBlockZKern(float* dst, int D2, float* src, int S1, int S2, int block){
  
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<S1 && j<S2){ // we are in the source array
	   dst[i*D2 + block*S2 + j] = src[i*S2 + j];
    }
}


void insertBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1); // range over source size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		for(int i=0; i<S0; i++){
			float* src2D = &(src[dev][i*S1Part*S2]);
			float* dst2D = &(dst[dev][i*S1Part*D2]); //D1==S1
			insertBlockZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst2D, D2, src2D, S1Part, S2, block);///@todo stream or loop in kernel
		}
	}
}


/// @internal Extracts a matrix ("block") from src, a larger matrix
/// The position of of the block in src is block*D2 along the N2 direction.
__global__ void extractBlockZKern(float* dst, int D1, int D2, float *src, int S2, int block){
  
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i<D1 && j<D2){ // we are in the destination array
     dst[i*D2 + j] = src[i*S2 + block*D2 + j];
    }
}


void extractBlockZAsync(float **dst, int D0, int D1Part, int D2, float **src, int S2, int block, CUstream *streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1Part, BLOCKSIZE), 1); // range over destination size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    for(int i=0; i<D0; i++){
      float* src2D = &(src[dev][i*D1Part*S2]);
      float* dst2D = &(dst[dev][i*D1Part*D2]); //D1==S1
      extractBlockZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst2D, D1Part, D2, src2D, S2, block);///@todo stream or loop in kernel
    }
  }
}





/// @internal Does Z-padding and unpadding of a 2D matrix.
///	Fills padding space with zeros.
__global__ void copyPad2dKern(float* dst, int D2, float* src, int S1, int S2){
  
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;
   int D1 = S1; ///@todo:rm

	// this check makes it work for padding as well as for unpadding.
	// 2 separate functions are probably not more efficient
	// due to memory bandwidth limitations
   if (i<D1 && j<D2){ // if we are in the destination array we should write something
		if(i<S1 && j<S2){ // we are in the source array: copy the source
		   dst[i*D2 + j] = src[i*S2 + j];
		}else{ // we are out of the source array: write zero
			dst[i*D2 + j] = 0.0f;	
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
		for(int i=0; i<S0; i++){
			float* src2D = &(src[dev][i*S1Part*S2]);
			float* dst2D = &(dst[dev][i*S1Part*D2]); //D1==S1
			copyPad2dKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst2D, D2, src2D, S1Part, S2);///@todo stream or loop in kernel
		}
	}
}


#ifdef __cplusplus
}
#endif

