// The code in this source file is based on the reduction code from the CUDPP library. Hence the following notice:

/*
Copyright (c) 2007-2010 The Regents of the University of California, Davis
campus ("The Regents") and NVIDIA Corporation ("NVIDIA"). All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the The Regents, nor NVIDIA, nor the names of its
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This code has been significantly modified from its original version by Arne Vansteenkiste, 2011.
//  - restricted to use only floats
//  - more reduction operations than the original "sum" have been added (min, max, maxabs, ...)
//  - added streams for asynchronous execution
// Note that you have to comply with both the above BSD and GPL licences.

//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

///@todo case 1024 should be added to take advantage of modern GPUs

#include "reduce.h"
#include "multigpu.h"
#include "gpu_safe.h"

extern "C"
bool isPow2(unsigned int x){
  return ((x&(x-1))==0);
}

/// @internal
/// Utility class used to avoid linker errors with extern
/// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
  __device__ inline operator       T*()
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T*() const
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

//________________________________________________________________________________________________________________ kernels


/// This kernel takes a partial sum
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_sum_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    mySum += g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i+blockSize];
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { mySum = mySum + sdata[tid + 256]; sdata[tid] = mySum; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { mySum = mySum + sdata[tid + 128]; sdata[tid] = mySum; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { mySum = mySum + sdata[tid +  64]; sdata[tid] = mySum; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { mySum = mySum + smem[tid + 32]; smem[tid] = mySum;  }
      if (blockSize >=  32) { mySum = mySum + smem[tid + 16]; smem[tid] = mySum;  }
      if (blockSize >=  16) { mySum = mySum + smem[tid +  8]; smem[tid] = mySum;  }
      if (blockSize >=   8) { mySum = mySum + smem[tid +  4]; smem[tid] = mySum;  }
      if (blockSize >=   4) { mySum = mySum + smem[tid +  2]; smem[tid] = mySum;  }
      if (blockSize >=   2) { mySum = mySum + smem[tid +  1]; smem[tid] = mySum;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}

/// This kernel takes a partial maximum
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_max_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMax = -6E38;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMax = fmax(myMax, g_idata[i]);
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMax = fmax(myMax, g_idata[i+blockSize]);
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMax;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMax = fmax(myMax, sdata[tid + 256]); sdata[tid] = myMax; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMax = fmax(myMax, sdata[tid + 128]); sdata[tid] = myMax; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMax = fmax(myMax, sdata[tid +  64]); sdata[tid] = myMax; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMax = fmax(myMax, smem[tid + 32]); smem[tid] = myMax;  }
      if (blockSize >=  32) { myMax = fmax(myMax, smem[tid + 16]); smem[tid] = myMax;  }
      if (blockSize >=  16) { myMax = fmax(myMax, smem[tid +  8]); smem[tid] = myMax;  }
      if (blockSize >=   8) { myMax = fmax(myMax, smem[tid +  4]); smem[tid] = myMax;  }
      if (blockSize >=   4) { myMax = fmax(myMax, smem[tid +  2]); smem[tid] = myMax;  }
      if (blockSize >=   2) { myMax = fmax(myMax, smem[tid +  1]); smem[tid] = myMax;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}


/// This kernel takes a partial minimum
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_min_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMin = 6E38;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMin = fmin(myMin, g_idata[i]);
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMin = fmin(myMin, g_idata[i+blockSize]);
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMin;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMin = fmin(myMin, sdata[tid + 256]); sdata[tid] = myMin; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMin = fmin(myMin, sdata[tid + 128]); sdata[tid] = myMin; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMin = fmin(myMin, sdata[tid +  64]); sdata[tid] = myMin; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMin = fmin(myMin, smem[tid + 32]); smem[tid] = myMin;  }
      if (blockSize >=  32) { myMin = fmin(myMin, smem[tid + 16]); smem[tid] = myMin;  }
      if (blockSize >=  16) { myMin = fmin(myMin, smem[tid +  8]); smem[tid] = myMin;  }
      if (blockSize >=   8) { myMin = fmin(myMin, smem[tid +  4]); smem[tid] = myMin;  }
      if (blockSize >=   4) { myMin = fmin(myMin, smem[tid +  2]); smem[tid] = myMin;  }
      if (blockSize >=   2) { myMin = fmin(myMin, smem[tid +  1]); smem[tid] = myMin;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}


/// This kernel takes a partial maximum of absolute values
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_maxabs_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMaxabs = 0.;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMaxabs = fmax(myMaxabs, fabs(g_idata[i]));
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMaxabs = fmax(myMaxabs, fabs(g_idata[i+blockSize]));
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMaxabs;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMaxabs = fmax(myMaxabs, sdata[tid + 256]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMaxabs = fmax(myMaxabs, sdata[tid + 128]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMaxabs = fmax(myMaxabs, sdata[tid +  64]); sdata[tid] = myMaxabs; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMaxabs = fmax(myMaxabs, smem[tid + 32]); smem[tid] = myMaxabs;  }
      if (blockSize >=  32) { myMaxabs = fmax(myMaxabs, smem[tid + 16]); smem[tid] = myMaxabs;  }
      if (blockSize >=  16) { myMaxabs = fmax(myMaxabs, smem[tid +  8]); smem[tid] = myMaxabs;  }
      if (blockSize >=   8) { myMaxabs = fmax(myMaxabs, smem[tid +  4]); smem[tid] = myMaxabs;  }
      if (blockSize >=   4) { myMaxabs = fmax(myMaxabs, smem[tid +  2]); smem[tid] = myMaxabs;  }
      if (blockSize >=   2) { myMaxabs = fmax(myMaxabs, smem[tid +  1]); smem[tid] = myMaxabs;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}



/// This kernel takes a partial maximum difference between two arrays
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_maxdiff_kernel(float* a, float* b, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMaxabs = 0.;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMaxabs = fmax(myMaxabs, fabs(a[i]-b[i]));
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMaxabs = fmax(myMaxabs, fabs(a[i+blockSize]-b[i+blockSize]));
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMaxabs;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMaxabs = fmax(myMaxabs, sdata[tid + 256]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMaxabs = fmax(myMaxabs, sdata[tid + 128]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMaxabs = fmax(myMaxabs, sdata[tid +  64]); sdata[tid] = myMaxabs; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMaxabs = fmax(myMaxabs, smem[tid + 32]); smem[tid] = myMaxabs;  }
      if (blockSize >=  32) { myMaxabs = fmax(myMaxabs, smem[tid + 16]); smem[tid] = myMaxabs;  }
      if (blockSize >=  16) { myMaxabs = fmax(myMaxabs, smem[tid +  8]); smem[tid] = myMaxabs;  }
      if (blockSize >=   8) { myMaxabs = fmax(myMaxabs, smem[tid +  4]); smem[tid] = myMaxabs;  }
      if (blockSize >=   4) { myMaxabs = fmax(myMaxabs, smem[tid +  2]); smem[tid] = myMaxabs;  }
      if (blockSize >=   2) { myMaxabs = fmax(myMaxabs, smem[tid +  1]); smem[tid] = myMaxabs;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}

//________________________________________________________________________________________________________________ kernel wrappers

#ifdef __cplusplus
extern "C" {
#endif

// single-GPU
void partialSumAsync1(float* d_idata, float* d_odata, int blocks, int threads, int size, CUstream stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_sum_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_sum_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_sum_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_sum_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_sum_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_sum_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_sum_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_sum_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_sum_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_sum_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_sum_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_sum_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_sum_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_sum_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_sum_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_sum_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_sum_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_sum_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_sum_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_sum_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
}

void partialSumAsync(float** input, float** output, int blocks, int threadsPerBlock, int N, CUstream* stream){
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		partialSumAsync1(input[dev], output[dev], blocks, threadsPerBlock, N, stream[dev]);
	}
}





// single-GPU
void partialMaxAsync1(float* d_idata, float* d_odata, int blocks, int threads, int size, CUstream stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_max_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_max_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_max_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_max_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_max_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_max_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_max_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_max_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_max_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_max_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_max_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_max_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_max_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_max_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_max_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_max_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_max_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_max_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_max_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_max_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
}


void partialMaxAsync(float** input, float** output, int blocks, int threadsPerBlock, int N, CUstream* stream){
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		partialMaxAsync1(input[dev], output[dev], blocks, threadsPerBlock, N, stream[dev]);
	}
}





void partialMinAsync1(float* d_idata, float* d_odata, int blocks, int threads, int size, CUstream stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_min_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_min_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_min_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_min_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_min_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_min_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_min_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_min_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_min_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_min_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_min_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_min_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_min_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_min_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_min_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_min_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_min_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_min_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_min_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_min_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
}


void partialMinAsync(float** input, float** output, int blocks, int threadsPerBlock, int N, CUstream* stream){
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		partialMinAsync1(input[dev], output[dev], blocks, threadsPerBlock, N, stream[dev]);
	}
}




// Single-GPU
void partialMaxAbsAsync1(float* d_idata, float* d_odata, int blocks, int threads, int size, CUstream stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_maxabs_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_maxabs_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_maxabs_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_maxabs_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_maxabs_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_maxabs_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_maxabs_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_maxabs_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_maxabs_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_maxabs_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_maxabs_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 256: _gpu_maxabs_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case 128: _gpu_maxabs_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  64: _gpu_maxabs_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  32: _gpu_maxabs_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case  16: _gpu_maxabs_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   8: _gpu_maxabs_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   4: _gpu_maxabs_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   2: _gpu_maxabs_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
      case   1: _gpu_maxabs_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
    }
  }
}


void partialMaxAbsAsync(float** input, float** output, int blocks, int threadsPerBlock, int N, CUstream* stream){
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		partialMaxAbsAsync1(input[dev], output[dev], blocks, threadsPerBlock, N, stream[dev]);
	}
}



// Single-GPU
void partialMaxDiffAsync1(float* a, float* b, float* d_odata, int blocks, int threads, int size, CUstream stream) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_maxdiff_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case 256: _gpu_maxdiff_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case 128: _gpu_maxdiff_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  64: _gpu_maxdiff_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  32: _gpu_maxdiff_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  16: _gpu_maxdiff_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   8: _gpu_maxdiff_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   4: _gpu_maxdiff_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   2: _gpu_maxdiff_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   1: _gpu_maxdiff_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_maxdiff_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case 256: _gpu_maxdiff_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case 128: _gpu_maxdiff_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  64: _gpu_maxdiff_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  32: _gpu_maxdiff_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case  16: _gpu_maxdiff_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   8: _gpu_maxdiff_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   4: _gpu_maxdiff_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   2: _gpu_maxdiff_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
      case   1: _gpu_maxdiff_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(a, b, d_odata, size); break;
    }
  }
}


void partialMaxDiffAsync(float** a, float** b, float** output, int blocks, int threadsPerBlock, int N, CUstream* stream){
	for (int dev = 0; dev < nDevice(); dev++) {
		gpu_safe(cudaSetDevice(deviceId(dev)));
		partialMaxDiffAsync1(a[dev], b[dev], output[dev], blocks, threadsPerBlock, N, stream[dev]);
	}
}

///// This kernel takes a partial sum of absolute values
//template <unsigned int blockSize, bool nIsPow2>
//__global__ void _gpu_sumabs_kernel(float* g_idata, float* g_odata, unsigned int n) {
//  float* sdata = SharedMemory<float>();
//
//  // perform first level of reduction,
//  // reading from global memory, writing to shared memory
//  unsigned int tid = threadIdx.x;
//  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
//  unsigned int gridSize = blockSize*2*gridDim.x;
//
//  float mySum = 0;
//
//  // we reduce multiple elements per thread.  The number is determined by the
//  // number of active thread blocks (via gridDim).  More blocks will result
//  // in a larger gridSize and therefore fewer elements per thread
//  while (i < n)
//  {
//    mySum += fabs(g_idata[i]);
//    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
//    if (nIsPow2 || i + blockSize < n)
//      mySum += fabs(g_idata[i+blockSize]);
//    i += gridSize;
//  }
//
//  // each thread puts its local sum into shared memory
//  sdata[tid] = mySum;
//  __syncthreads();
//
//
//  // do reduction in shared mem
//  if (blockSize >= 512) { if (tid < 256) { mySum = mySum + sdata[tid + 256]; sdata[tid] = mySum; } __syncthreads(); }
//  if (blockSize >= 256) { if (tid < 128) { mySum = mySum + sdata[tid + 128]; sdata[tid] = mySum; } __syncthreads(); }
//  if (blockSize >= 128) { if (tid <  64) { mySum = mySum + sdata[tid +  64]; sdata[tid] = mySum; } __syncthreads(); }
//
//  if (tid < 32)
//    {
//      // now that we are using warp-synchronous programming (below)
//      // we need to declare our shared memory volatile so that the compiler
//      // doesn't reorder stores to it and induce incorrect behavior.
//      volatile float* smem = sdata;
//      if (blockSize >=  64) { mySum = mySum + smem[tid + 32]; smem[tid] = mySum;  }
//      if (blockSize >=  32) { mySum = mySum + smem[tid + 16]; smem[tid] = mySum;  }
//      if (blockSize >=  16) { mySum = mySum + smem[tid +  8]; smem[tid] = mySum;  }
//      if (blockSize >=   8) { mySum = mySum + smem[tid +  4]; smem[tid] = mySum;  }
//      if (blockSize >=   4) { mySum = mySum + smem[tid +  2]; smem[tid] = mySum;  }
//      if (blockSize >=   2) { mySum = mySum + smem[tid +  1]; smem[tid] = mySum;  }
//    }
//    // write result for this block to global mem
//    if (tid == 0)
//      g_odata[blockIdx.x] = sdata[0];
//}
//
//void gpu_partial_sumabs(float* d_idata, float* d_odata, int blocks, int threads, int size) {
//  dim3 dimBlock(threads, 1, 1);
//  dim3 dimGrid(blocks, 1, 1);
//
//  // when there is only one warp per block, we need to allocate two warps
//  // worth of shared memory so that we don't index shared memory out of bounds
//  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
//
//  if (isPow2(size))
//  {
//    switch (threads)
//    {
//      case 512: _gpu_sumabs_kernel<512, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case 256: _gpu_sumabs_kernel<256, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case 128: _gpu_sumabs_kernel<128, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  64: _gpu_sumabs_kernel< 64, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  32: _gpu_sumabs_kernel< 32, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  16: _gpu_sumabs_kernel< 16, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   8: _gpu_sumabs_kernel<  8, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   4: _gpu_sumabs_kernel<  4, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   2: _gpu_sumabs_kernel<  2, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   1: _gpu_sumabs_kernel<  1, true><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//    }
//  }
//  else
//  {
//    switch (threads)
//    {
//      case 512: _gpu_sumabs_kernel<512, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case 256: _gpu_sumabs_kernel<256, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case 128: _gpu_sumabs_kernel<128, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  64: _gpu_sumabs_kernel< 64, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  32: _gpu_sumabs_kernel< 32, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case  16: _gpu_sumabs_kernel< 16, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   8: _gpu_sumabs_kernel<  8, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   4: _gpu_sumabs_kernel<  4, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   2: _gpu_sumabs_kernel<  2, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//      case   1: _gpu_sumabs_kernel<  1, false><<< dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size); break;
//    }
//  }
//}


#ifdef __cplusplus
}
#endif
