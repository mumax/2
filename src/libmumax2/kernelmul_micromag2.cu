/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "kernelmul_micromag2.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The size of matrix blocks to be loaded into shared memory.  BLOCKSIZE_J can be made smaller if needed, BLOCKSIZE_K better not.
#define BLOCKSIZE_K 16
#define BLOCKSIZE_J 4  

/// |Hx|   |Kxx Kxy Kxz|   |Mx|
/// |Hy| = |Kxy Kyy Kyz| * |My|
/// |Hz|   |Kxz Kyz Kzz|   |Mz|
__global__ void kernelMulMicromag3D2Kern(
    float* fftMx,  float* fftMy, float* fftMz, 
    float* fftKxx, float* fftKyy, float* fftKzz,
    float* fftKyz, float* fftKxz, float* fftKxy, int N0, int N1, int N2){
  
  int index, k_index;
  float Mx, My, Mz;

  __shared__ float Kxx[BLOCKSIZE_J][BLOCKSIZE_K/2+1];
  __shared__ float Kxy[BLOCKSIZE_J][BLOCKSIZE_K/2+1];
  __shared__ float Kxz[BLOCKSIZE_J][BLOCKSIZE_K/2+1];
  __shared__ float Kyy[BLOCKSIZE_J][BLOCKSIZE_K/2+1];
  __shared__ float Kyz[BLOCKSIZE_J][BLOCKSIZE_K/2+1];
  __shared__ float Kzz[BLOCKSIZE_J][BLOCKSIZE_K/2+1];

  // index of the block inside the blockmatrix
  int Bk = blockIdx.x;  
  int Bj = blockIdx.y;

  // "minor" indices inside the tile
  int k = threadIdx.x;
  int j = threadIdx.y;

  // index in the total array
  int K = Bk*BLOCKSIZE_K + k;
  int J = Bj*BLOCKSIZE_J + j;

  int kmax = 0;
  if ( (N2/4 - (Bk+1)*BLOCKSIZE_K/2)>=0 )
    kmax = BLOCKSIZE_K/2;
  else
    kmax =  (Bk+1)*BLOCKSIZE_K/2 - N2/4 ;
 
  for (int i=0; i<N0/2+1; i++){

    // Copying kernel components to shared memory ------------------------------------
    int N2K = N2/2;        //TODO: delete this line when reduced storage of kernel is implemented 
    //  int N2K = N2/4+1;  //TODO: use this line when reduced storage of kernel is implemented 
    
    if (J<N1 && K<(N2/2+1) && k<BLOCKSIZE_K/2+1) {
      index = i*N1*N2K + J*N2K + Bk*BLOCKSIZE_K/2 + k;
      Kxx[j][k] = fftKxx[index];
      Kxy[j][k] = fftKxy[index];
      Kxz[j][k] = fftKxz[index];
      Kyy[j][k] = fftKyy[index];
      Kyz[j][k] = fftKyz[index];
      Kzz[j][k] = fftKzz[index];
    }
    __syncthreads();
    // -------------------------------------------------------------------------------
    
  
    // Perform kernel multiplication -------------------------------------------------
    if ( J<N1 && (K<N2/2) ) {

      index = i*N1*N2 + J*N2 + K;
      Mx = fftMx[index];
      My = fftMy[index];
      Mz = fftMz[index];
      fftMx[index] = Kxx[j][k/2]*Mx + Kxy[j][k/2]*My + Kxz[j][k/2]*Mz;
      fftMy[index] = Kxy[j][k/2]*Mx + Kyy[j][k/2]*My + Kyz[j][k/2]*Mz;
      fftMz[index] = Kxz[j][k/2]*Mx + Kyz[j][k/2]*My + Kzz[j][k/2]*Mz;

      index = i*N1*N2 + J*N2 + N2 - Bk*BLOCKSIZE_K - 2*kmax + k;
      Mx = fftMx[index];
      My = fftMy[index];
      Mz = fftMz[index];
      k_index = kmax-k/2;
      fftMx[index] = Kxx[j][k_index]*Mx + Kxy[j][k_index]*My + Kxz[j][k_index]*Mz;
      fftMy[index] = Kxy[j][k_index]*Mx + Kyy[j][k_index]*My + Kyz[j][k_index]*Mz;
      fftMz[index] = Kxz[j][k_index]*Mx + Kyz[j][k_index]*My + Kzz[j][k_index]*Mz;
      
      if (i!=0 && i!=(N0/2+1)){
        index = (N0-i)*N1*N2 + J*N2 + K;
        Mx = fftMx[index];
        My = fftMy[index];
        Mz = fftMz[index];
        fftMx[index] = Kxx[j][k/2]*Mx + Kxy[j][k/2]*My + Kxz[j][k/2]*Mz;
        fftMy[index] = Kxy[j][k/2]*Mx + Kyy[j][k/2]*My + Kyz[j][k/2]*Mz;
        fftMz[index] = Kxz[j][k/2]*Mx + Kyz[j][k/2]*My + Kzz[j][k/2]*Mz;
        
        index = (N0-i)*N1*N2 + J*N2 + N2 - Bk*BLOCKSIZE_K - 2*kmax + k;
        Mx = fftMx[index];
        My = fftMy[index];
        Mz = fftMz[index];
        k_index = kmax-k/2;
        fftMx[index] = Kxx[j][k_index]*Mx + Kxy[j][k_index]*My + Kxz[j][k_index]*Mz;
        fftMy[index] = Kxy[j][k_index]*Mx + Kyy[j][k_index]*My + Kyz[j][k_index]*Mz;
        fftMz[index] = Kxz[j][k_index]*Mx + Kyz[j][k_index]*My + Kzz[j][k_index]*Mz;
      }
      __syncthreads();    
    // -------------------------------------------------------------------------------
    }
    
  }
  
  return;
}

void kernelMulMicromag3D2Async(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz,
                              float** fftKyz, float** fftKxz, float** fftKxy,
                              CUstream* stream, int* partSize){

  // based on sizes of the sources as they are stored on 1 GPU after transpose
  int N0 = partSize[0];
  int N1 = partSize[1];
  int N2 = partSize[2];

  printf("\n\n%d, %d\n\n",(N2/2-1) / BLOCKSIZE_K + 1, (N1-1) / BLOCKSIZE_J + 1);
  //N2 devided by 2 since symmetry in second half is exlpoited (except for 1 element)
  dim3 gridsize((N2/2-1) / BLOCKSIZE_K + 1, (N1-1) / BLOCKSIZE_J + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
  dim3 blocksize(BLOCKSIZE_K, BLOCKSIZE_J, 1);
  

  for (int dev = 0; dev < nDevice(); dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
//    kernelMulMicromag3D2Kern<<<gridsize, blocksize, 0,cudaStream_t(stream[dev])>>>
    kernelMulMicromag3D2Kern<<<gridsize, blocksize, 0,stream[dev]>>>
      ( fftMx[dev],  fftMy[dev],  fftMz[dev],
        fftKxx[dev], fftKyy[dev], fftKzz[dev],
        fftKyz[dev], fftKxz[dev], fftKxy[dev], 
        N0, N1, N2);
	}
}




//// |Hx|   |Kxx  0   0 |   |Mx|
//// |Hy| = | 0  Kyy Kyz| * |My|
//// |Hz|   | 0  Kyz Kzz|   |Mz|
//
//__global__ void _gpu_kernelmul4(float* fftMx,  float* fftMy,  float* fftMz,
//                                float* fftKxx, float* fftKyy, float* fftKzz, float* fftKyz, int N){
//  int i = threadindex;
//  int e = 2 * i;
//
//  // we some shared memory here, which saves an "8N" buffer in the global memory
//  ///@todo coalescale read/writes, cleanup indices
//  if(i < N){
//  float reMx = fftMx[e  ];
//  float imMx = fftMx[e+1];
//
//  float reMy = fftMy[e  ];
//  float imMy = fftMy[e+1];
//
//  float reMz = fftMz[e  ];
//  float imMz = fftMz[e+1];
//
//  float Kxx = fftKxx[i];
//  float Kyy = fftKyy[i];
//  float Kyz = fftKyz[i];
//  float Kzz = fftKzz[i];
//  
//  fftMx[e  ] = reMx * Kxx;
//  fftMx[e+1] = imMx * Kxx;
//  fftMy[e  ] = reMy * Kyy + reMz * Kyz;
//  fftMy[e+1] = imMy * Kyy + imMz * Kyz;
//  fftMz[e  ] = reMy * Kyz + reMz * Kzz;
//  fftMz[e+1] = imMy * Kyz + imMz * Kzz;
//  }
//  
//  return;
//}
//
//void gpu_kernelmul4(float *fftMx, float *fftMy, float *fftMz, 
//                    float *fftKxx, float *fftKyy, float *fftKzz, float *fftKyz, 
//                    int nRealNumbers){
//
//  //timer_start("kernel_mul");
//  assert(nRealNumbers > 0);
//  assert(nRealNumbers % 2 == 0);
//
//  dim3 gridSize, blockSize;
//  make1dconf(nRealNumbers/2, &gridSize, &blockSize);
//
//  _gpu_kernelmul4<<<gridSize, blockSize>>>(fftMx, fftMy, fftMz, fftKxx, fftKyy, fftKzz, fftKyz, nRealNumbers/2);
//  gpu_sync();
//  //timer_stop("kernel_mul");
// 
//  return;
//}
//
//
//
//// |Hx|   | 0  0   0 |   |Mx|
//// |Hy| = | 0 Kyy Kyz| * |My|
//// |Hz|   | 0 Kyz Kzz|   |Mz|
//
//__global__ void _gpu_kernelmul3(float* fftMy,  float* fftMz,
//                                float* fftKyy, float* fftKzz, float* fftKyz, int N){
//  int i = threadindex;
//  int e = 2 * i;
//
//  // we some shared memory here, which saves an "8N" buffer in the global memory
//  ///@todo coalescale read/writes, cleanup indices
//  if(i < N){
//
//  float reMy = fftMy[e  ];
//  float imMy = fftMy[e+1];
//
//  float reMz = fftMz[e  ];
//  float imMz = fftMz[e+1];
//
//  float Kyy = fftKyy[i];
//  float Kyz = fftKyz[i];
//  float Kzz = fftKzz[i];
//  
//  fftMy[e  ] = reMy * Kyy + reMz * Kyz;
//  fftMy[e+1] = imMy * Kyy + imMz * Kyz;
//  fftMz[e  ] = reMy * Kyz + reMz * Kzz;
//  fftMz[e+1] = imMy * Kyz + imMz * Kzz;
//  }
//  
//  return;
//}
//
//void gpu_kernelmul3(float *fftMy, float *fftMz, 
//                    float *fftKyy, float *fftKzz, float *fftKyz, 
//                    int nRealNumbers){
//
//  //timer_start("kernel_mul");
//  assert(nRealNumbers > 0);
//  assert(nRealNumbers % 2 == 0);
//
//  dim3 gridSize, blockSize;
//  make1dconf(nRealNumbers/2, &gridSize, &blockSize);
//
//  _gpu_kernelmul3<<<gridSize, blockSize>>>(fftMy, fftMz, fftKyy, fftKzz, fftKyz, nRealNumbers/2);
//  gpu_sync();
//  //timer_stop("kernel_mul");
// 
//  return;
//}
//
//
//
//// |Hx|   | 0   Kz -Ky|   |Jx|
//// |Hy| = |-Kz  0   Kx| * |Jy|
//// |Hz|   | Ky -Kx  0 |   |Jz|
//
//__global__ void _gpu_kernelmul_biot_savart3D(float* fftJx,  float* fftJy,  float* fftJz,
//                                             float* fftKx, float* fftKy, float* fftKz,
//                                             int N){
//  int i = threadindex;
//  int e = 2 * i;
//
//  // we some shared memory here, which saves an "8N" buffer in the global memory
//  if(i < N){
//    float reJx = fftJx[e  ];
//    float imJx = fftJx[e+1];
//
//    float reJy = fftJy[e  ];
//    float imJy = fftJy[e+1];
//
//    float reJz = fftJz[e  ];
//    float imJz = fftJz[e+1];
//
//    float Kx = fftKx[i];
//    float Ky = fftKy[i];
//    float Kz = fftKz[i];
//    
//    fftJx[e  ] =  reJy * Kz - reJz * Ky;
//    fftJx[e+1] =  imJy * Kz - imJz * Ky;
//
//    fftJy[e  ] = -reJx * Kz + reJz * Kx;
//    fftJy[e+1] = -imJx * Kz + imJz * Kx;
//
//    fftJz[e  ] =  reJx * Ky - reJy * Kx;
//    fftJz[e+1] =  imJx * Ky - imJy * Kx;
//  }
//  
//  return;
//}
//
//void gpu_kernelmul_biot_savart3D(float* fftJx, float* fftJy, float* fftJz,
//                                 float* fftKx, float* fftKy, float* fftKz,
//                                 int nRealNumbers){
//
//  //timer_start("kernel_mul");
//  assert(nRealNumbers > 0);
//  assert(nRealNumbers % 2 == 0);
//
//  dim3 gridSize, blockSize;
//  make1dconf(nRealNumbers/2, &gridSize, &blockSize);
//
//  _gpu_kernelmul_biot_savart3D<<<gridSize, blockSize>>>(fftJx, fftJy, fftJz,
//                                           fftKx, fftKy, fftKz,
//                                           nRealNumbers/2);
//  gpu_sync();
//  //timer_stop("kernel_mul");
//  
//  return;
//}
//
//
//
//// |Hx|   | 0   Kz -Ky|   |Jx|
//// |Hy| = |-Kz  0   0 | * |Jy|
//// |Hz|   | Ky  0   0 |   |Jz|
//
//__global__ void _gpu_kernelmul_biot_savart3D_Nx1(float* fftJx, float* fftJy, float* fftJz,
//                                                 float* fftKy, float* fftKz,
//                                                 int N){
//  int i = threadindex;
//  int e = 2 * i;
//
//  // we some shared memory here, which saves an "8N" buffer in the global memory
//  if(i < N){
//    float reJx = fftJx[e  ];
//    float imJx = fftJx[e+1];
//
//    float reJy = fftJy[e  ];
//    float imJy = fftJy[e+1];
//
//    float reJz = fftJz[e  ];
//    float imJz = fftJz[e+1];
//
//    float Ky = fftKy[i];
//    float Kz = fftKz[i];
//    
//    fftJx[e  ] =  reJy * Kz - reJz * Ky;
//    fftJx[e+1] =  imJy * Kz - imJz * Ky;
//
//    fftJy[e  ] = -reJx * Kz;
//    fftJy[e+1] = -imJx * Kz;
//
//    fftJz[e  ] =  reJx * Ky;
//    fftJz[e+1] =  imJx * Ky;
//  }
//  
//  return;
//}
//
//void gpu_kernelmul_biot_savart3DNx1(float* fftJx, float* fftJy, float* fftJz,
//                                    float* fftKy, float* fftKz,
//                                    int nRealNumbers){
//
//  //timer_start("kernel_mul");
//  assert(nRealNumbers > 0);
//  assert(nRealNumbers % 2 == 0);
//
//  dim3 gridSize, blockSize;
//  make1dconf(nRealNumbers/2, &gridSize, &blockSize);
//
//  _gpu_kernelmul_biot_savart3D_Nx1<<<gridSize, blockSize>>>(fftJx, fftJy, fftJz, fftKy, fftKz, nRealNumbers/2);
//  gpu_sync();
//  //timer_stop("kernel_mul");
//  
//  return;
//}
//
//
//
//// |Hx|   | 0   0  0|   |Jx|
//// |Hy| = |-Kz  0  0| * | 0|
//// |Hz|   | Ky  0  0|   | 0|
//
//__global__ void _gpu_kernelmul_biot_savart2D(float* fftJx,  float* fftJy,  float* fftJz,
//                                             float* fftKy, float* fftKz,
//                                             int N){
//  int i = threadindex;
//  int e = 2 * i;
//
//  // we some shared memory here, which saves an "8N" buffer in the global memory
//  if(i < N){
//    float reJx = fftJx[e  ];
//    float imJx = fftJx[e+1];
//
//    float Ky = fftKy[i];
//    float Kz = fftKz[i];
//    
//    fftJy[e  ] = -reJx * Kz;
//    fftJy[e+1] = -imJx * Kz;
//
//    fftJz[e  ] =  reJx * Ky;
//    fftJz[e+1] =  imJx * Ky;
//  }
//  
//  return;
//}
//
//void gpu_kernelmul_biot_savart2D(float* fftJx,  float* fftJy,  float* fftJz,
//                                 float* fftKy, float* fftKz,
//                                 int nRealNumbers){
//
//  //timer_start("kernel_mul");
//  assert(nRealNumbers > 0);
//  assert(nRealNumbers % 2 == 0);
//
//  dim3 gridSize, blockSize;
//  make1dconf(nRealNumbers/2, &gridSize, &blockSize);
//
//  _gpu_kernelmul_biot_savart2D<<<gridSize, blockSize>>>(fftJx, fftJy, fftJz, fftKy, fftKz, nRealNumbers/2);
//  gpu_sync();
//  //timer_stop("kernel_mul");
//  
//  return;
//}



#ifdef __cplusplus
}
#endif
