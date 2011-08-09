//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file implements GPU math functions

#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif


/// For the first N elements, dst[i] = a[i] + b[i]
__global__ void gpuAdd(float* dst, float* a, float* b, int N){
	int i = threadindex;
	if(i < N){
		dst[i] = a[i] + b[i];
	}
}


#ifdef __cplusplus
}
#endif
