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

/// multi-GPU Partial sum function.
void partialSumsAsync(float** input,      ///< input data. size = N 
                    float** output,       ///< partially reduced data, usually reduced further on CPU. size = blocks
                    int blocks,          ///< patially reduce in X blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
                    int threadsPerBlock, ///< use X threads per block: @warning must be < N
                    int N,               ///< size of input data, must be > threadsPerBlock
					CUstream** stream    ///< cuda stream for async execution
                    );


#ifdef __cplusplus
}
#endif
#endif
