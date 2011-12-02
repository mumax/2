//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

/// This file implements various functions used for debugging.

#include "index.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif


/// @debug sets array[i,j,k] to its C-oder X (outer) index.
__global__ void setIndexXKern(float* dst, int PART, int N0, int N1Part, int N2){

  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //float j2 = j + PART * N1Part; // j-index in the big array
  if (j < N1Part && k < N2){
	for(int i=0; i<N0; i++){
  		int I = i*N1Part*N2 + j*N2 + k; // linear array index
			dst[I] = i; 
		}
	}
}



void setIndexX(float** dst, int N0, int N1Part, int N2) {
	dim3 gridSize, blockSize;
	make2dconf(N1Part, N2, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		setIndexXKern <<<gridSize, blockSize>>> (dst[dev], dev, N0, N1Part, N2);
	}
}



/// @debug sets array[i,j,k] to its C-oder Y index.
__global__ void setIndexYKern(float* dst, int PART, int N0, int N1Part, int N2){

  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  float j2 = j + PART * N1Part; // j-index in the big array
  if (j < N1Part && k < N2){
	for(int i=0; i<N0; i++){
  		int I = i*N1Part*N2 + j*N2 + k; // linear array index
			dst[I] = j2; 
		}
	}
}



void setIndexY(float** dst, int N0, int N1Part, int N2) {
	dim3 gridSize, blockSize;
	make2dconf(N1Part, N2, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		setIndexYKern <<<gridSize, blockSize>>> (dst[dev], dev, N0, N1Part, N2);
	}
}



/// @debug sets array[i,j,k] to its C-oder Z (inner) index.
__global__ void setIndexZKern(float* dst, int PART, int N0, int N1Part, int N2){

  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //float j2 = j + PART * N1Part; // j-index in the big array
  if (j < N1Part && k < N2){
	for(int i=0; i<N0; i++){
  		int I = i*N1Part*N2 + j*N2 + k; // linear array index
			dst[I] = k; 
		}
	}
}



void setIndexZ(float** dst, int N0, int N1Part, int N2) {
	dim3 gridSize, blockSize;
	make2dconf(N1Part, N2, &gridSize, &blockSize);
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		setIndexZKern <<<gridSize, blockSize>>> (dst[dev], dev, N0, N1Part, N2);
	}
}



#ifdef __cplusplus
}
#endif


