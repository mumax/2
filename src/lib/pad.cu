#include "pad.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @author Arne Vansteenkiste, okt 2011



__global__ void copyBlockZKern(float* dst, int D2, float* src, int S1, int S2, int block){
  
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<S1 && j<S2){ // we are in the source array
	   dst[i*D2 + block*S2 + j] = src[i*S2 + j];
    }
}

void copyBlockZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, int block, CUstream* streams){

#define BLOCKSIZE 16 ///@todo use device properties

  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1); // range over source size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		for(int i=0; i<S0; i++){
			float* src2D = &(src[dev][i*S1Part*S2]);
			float* dst2D = &(dst[dev][i*S1Part*D2]); //D1==S1
			copyBlockZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst2D, D2, src2D, S1Part, S2, block);///@todo stream or loop in kernel
		}
	}
}



//__global__ void combineZKern(float* dst, int D2, float* src1, float* src2, int S1, int S2){
//  
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   int D1 = S1; ///@todo:rm
//
//	// this check makes it work for padding as well as for unpadding.
//	// 2 separate functions are probably not more efficient
//	// due to memory bandwidth limitations
//   if (i<D1 && j<D2){ // if we are in the destination array we should write something
//		if(i<S1){ // probably redundant
//			if(j<S2){ 	// we are in the source1 array
//		   		dst[i*D2 + j] = src1[i*S2 + j];
//			} else if(j<2*S2){ // we are in the source2 array
//				dst[i*D2 + j] = src2[i*S2 + (j-S2)];
//			}
//		}
//   }
//}
//
//
//void combineZAsync(float** dst, int D2, float** src1, float** src2, int S0, int S1Part, int S2, CUstream* streams){
//
//#define BLOCKSIZE 16 ///@todo use device properties
//
//  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1); // range over destination size
//  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//  check3dconf(gridSize, blockSize);
//
//	for (int dev = 0; dev < nDevice(); dev++) {
//		gpu_safe(cudaSetDevice(deviceId(dev)));
//		for(int i=0; i<S0; i++){
//			float* src1_2D = &(src1[dev][i*S1Part*S2]);
//			float* src2_2D = &(src2[dev][i*S1Part*S2]);
//			float* dst2D = &(dst[dev][i*S1Part*D2]); //D1==S1
//			combineZKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst2D, D2, src1_2D, src2_2D, S1Part, S2);///@todo stream or loop in kernel
//		}
//	}
//}



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

