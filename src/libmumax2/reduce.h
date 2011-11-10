//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


/**
 * @file reduce.h
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef REDUCE_H
#define REDUCE_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif


/// Multi-GPU partial sum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution                                         
void partialSumAsync(float** input, float** output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream* streams);


/// Multi-GPU partial maximum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution                                         
void partialMaxAsync(float** input, float** output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream* streams);


/// Multi-GPU partial minimum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution                                         
void partialMinAsync(float** input, float** output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream* streams);


/// Multi-GPU partial maximum of absolute values.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution                                         
void partialMaxAbsAsync(float** input, float** output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream* streams);


/// Multi-GPU partial maximum difference between arrays (max(abs(a[i]-b[i]))).
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution                                         
void partialMaxDiffAsync(float** a, float** b, float** output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream* streams);


#ifdef __cplusplus
}
#endif
#endif
