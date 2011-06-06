//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

/// This file implements various functions used for debugging.

#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @debug sets array[i] to i.
__global__ void SetIndex1D(float* part, int PART, int N){

  int i = threadindex;
  if (i < N){
	part[i] = i + PART*N;
  }
}



/// @debug sets array[i,j,k] to its C-oder index.
__global__ void SetIndex3D(float* part, int PART, int N0, int N1, int N2){

  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N1 && k < N2){
	for(int i=0; i<N0; i++){
  		int I = i*N1*N2 + j*N2 + k; // linear array index
			j += PART * N1;
			part[I] = i*1000 + j + k/1000;
		}
	}
}


#ifdef __cplusplus
}
#endif
