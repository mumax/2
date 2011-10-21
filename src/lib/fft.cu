#include "fft.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


/// @internal Does Z-padding and unpadding of a 2D matrix.
__global__ void copyPad2dKern(float* dst, int D2, float* src, int S1, int S2){
  
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

	// this check makes it work for padding as well as for unpadding.
	// 2 separate functions are probably not more efficient
	// due to memory bandwidth limitations
   if (i<S1 && j<S2 && j<D2){  // && i<D1: always true
		dst[i*D2 + j] = src[i*S2 + j];
   }
}

void copyPadZAsync(float** dst, int D2, float** src, int S0, int S1Part, int S2, CUstream* streams){
	assert(S2 <= D2);

#define BLOCKSIZE 16 ///@todo use device properties
  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1Part, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		//for  
			//&source[i*S1*S2],
		copyPad2dKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> (dst[dev], D2, src[dev], S1Part, S2);
	}
}



#ifdef __cplusplus
}
#endif

