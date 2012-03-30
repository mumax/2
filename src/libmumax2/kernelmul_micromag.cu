/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "kernelmul_micromag.h"
#include "multigpu.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif


/// |Hx|   |Kxx Kxy Kxz|   |Mx|
/// |Hy| = |Kxy Kyy Kyz| * |My|
/// |Hz|   |Kxz Kyz Kzz|   |Mz|
__global__ void kernelMulMicromag3DKern(float* fftMx,  float* fftMy,  float* fftMz,
                                        float* fftKxx, float* fftKyy, float* fftKzz,
                                        float* fftKyz, float* fftKxz, float* fftKxy, int N){
  int i = threadindex;
  int e = 2 * i;

  // we use some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes, cleanup indices
  if(i < N){
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kxx = fftKxx[i];
    float Kyy = fftKyy[i];
    float Kzz = fftKzz[i];

    float Kyz = fftKyz[i];
    float Kxz = fftKxz[i];
    float Kxy = fftKxy[i];

    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
//     fftMx[e  ] = (int)Kyy;
//     fftMx[e+1] = (int)Kyy;

    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
//     fftMy[e  ] = (int) Kyz;
//     fftMy[e+1] = (int) Kyz;

    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
/*    fftMz[e  ] = (int) Kzz;
    fftMz[e+1] = (int) Kzz;*/
  }
  
  return;
}

__export__ void kernelMulMicromag3DAsync(float** fftMx,  float** fftMy,  float** fftMz,
                              float** fftKxx, float** fftKyy, float** fftKzz,
                              float** fftKyz, float** fftKxz, float** fftKxy,
                              CUstream* stream, int partLen3D){

  assert(partLen3D > 0);
  assert(partLen3D % 2 == 0);

  dim3 gridSize, blockSize;
  make1dconf(partLen3D/2, &gridSize, &blockSize);

  for (int dev = 0; dev < nDevice(); dev++) {
	gpu_safe(cudaSetDevice(deviceId(dev)));
    kernelMulMicromag3DKern<<<gridSize, blockSize, 0, cudaStream_t(stream[dev])>>>( fftMx[dev],  fftMy[dev],  fftMz[dev],
                                                                                   fftKxx[dev], fftKyy[dev], fftKzz[dev],
                                                                                   fftKyz[dev], fftKxz[dev], fftKxy[dev], partLen3D/2);
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
