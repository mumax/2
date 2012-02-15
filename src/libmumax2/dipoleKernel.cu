#include "dipoleKernel.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @author Ben Van de Wiele

#define BLOCKSIZE 16 ///@todo use device properties

__device__ float getKernelElement(int N0, int N1, int N2, int co1, int co2, int a, int b, int c, int per0, int per1, int per2, 
                                  float cellX, float cellY, float cellZ, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  
  int cutoff = 400;    //square of the cutoff where the interaction is computed for dipole in the center in stead of magnetized volume

  
  // for elements in Kernel component gxx _________________________________________________________
    if (co1==0 && co2==0){

      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float x1 = (i + 0.5f) * cellX;
          float x2 = (i - 0.5f) * cellX;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellY + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellY * cellZ / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ *
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (i*cellX) * (i*cellX) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxy _________________________________________________________
    if (co1==0 && co2==1){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float x1 = (i + 0.5f) * cellX;
          float x2 = (i - 0.5f) * cellX;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellY + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellY * cellZ / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (- 3.0f* (i*cellX) * (j*cellY) * __powf(r2,-2.5f));
        }

      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyx (should be same result as gxy) __________________________
    if (co1==1 && co2==0){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float y1 = (j + 0.5f) * cellY;
          float y2 = (j - 0.5f) * cellY;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellX * cellZ / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( x*__powf(x*x+y1*y1+z*z, -1.5f) - x*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (- 3.0f* (i*cellX) * (j*cellY) * __powf(r2,-2.5f));
        }

      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxz _________________________________________________________
    if (co1==0 && co2==2){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float x1 = (i + 0.5f) * cellX;
          float x2 = (i - 0.5f) * cellX;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellY + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellY * cellZ / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (- 3.0f* (i*cellX) * (k*cellY) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzx (should be same result as gxz) __________________________
    if (co1==2 && co2==0){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float z1 = (k + 0.5f) * cellZ;
          float z2 = (k - 0.5f) * cellZ;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = k * cellY + dev_qd_P_10_Y[cnt2];
              result += cellX * cellY / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( x*__powf(x*x+y*y+z1*z1, -1.5f) - x*__powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (- 3.0f* (i*cellX) * (k*cellY) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float y1 = (j + 0.5f) * cellY;
          float y2 = (j - 0.5f) * cellY;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellX * cellZ / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (j*cellY) * (j*cellY) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyz _________________________________________________________
    if (co1==1 && co2==2){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float y1 = (j + 0.5f) * cellY;
          float y2 = (j - 0.5f) * cellY;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellZ + dev_qd_P_10_Z[cnt3];
              result += cellX * cellZ / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    ( - 3.0f* (j*cellY) * (k*cellZ) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzy _________________________________________________________
    if (co1==1 && co2==2){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float z1 = (k + 0.5f) * cellZ;
          float z2 = (k - 0.5f) * cellZ;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellY + dev_qd_P_10_Y[cnt2];
              result += cellX * cellY / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( y*__powf(x*x+y*y+z1*z1, -1.5f) - y*__powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    ( - 3.0f* (j*cellY) * (k*cellZ) * __powf(r2,-2.5f));
        }
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzz _________________________________________________________
    if (co1==2 && co2==2){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          float z1 = (k + 0.5f) * cellZ;
          float z2 = (k - 0.5f) * cellZ;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellY + dev_qd_P_10_Y[cnt2];
              result += cellX * cellY / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (k*cellZ) * (k*cellZ) * __powf(r2,-2.5f));
        }
       
      }
    }
  // ______________________________________________________________________________________________
  
  result *= -1.0f/4.0f/3.14159265f;
  return( result );
}





__global__ void initFaceKernel6ElementKern (float *data, int co1, int co2, 
                                            int N0, int N1, int N2, int N1part,
                                            int per0, int per1, int per2,
                                            float cellX, float cellY, float cellZ,
                                            float *dev_qd_P_10, float *dev_qd_W_10, 
                                            int dev, int NDev){

  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j2 = dev*N1part+j;

  int N12 = N1part*N2;

  if (j<N1part && k<N2/2){              

    for (int i=0; i<(N0+1)/2; i++){     // this also works in the 2D case
      if (j2<N1/2){
          data[i*N12 + j*N2 + k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0)
          data[(N0-i)*N12 + j*N2 + k] = 
            getKernelElement(N0, N1, N2, co1, co2, -i, j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (k>0)
          data[i*N12 + j*N2 + N2-k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0 && k>0)
          data[(N0-i)*N12 + j*N2 + N2-k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
      }
      if (j2>N1/2){
          data[i*N12 + j*N2 + k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, N1-j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0)
          data[(N0-i)*N12 + j*N2 + k] = 
            getKernelElement(N0, N1, N2, co1, co2, -i, N1-j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (k>0)
          data[i*N12 + j*N2 + N2-k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, N1-j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0 && k>0)
          data[(N0-i)*N12 + j*N2 + N2-k] = 
            getKernelElement(N0, N1, N2, co1, co2, i, N1-j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
      }
    }
    
  }
  
  return;
}



void initFaceKernel6ElementAsync(float **data, int co1, int co2,            /// data array and component
                                 int N0, int N1, int N2, int N1part,        /// size of the kernel
                                 int per0, int per1, int per2,              /// periodicity
                                 float cellX, float cellY, float cellZ,     /// cell size
                                 float **dev_qd_P_10, float **dev_qd_W_10,  /// quadrature points and weights
                                 CUstream *streams
                                ){

  dim3 gridSize(divUp(N2/2, BLOCKSIZE), divUp(N1part, BLOCKSIZE), 1); // range over destination size
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);
  
  int NDev = nDevice();
  for (int dev = 0; dev < NDev; dev++) {
    gpu_safe(cudaSetDevice(deviceId(dev)));
    initFaceKernel6ElementKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> 
      (data[dev], co1, co2, N0, N1, N2, N1part, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10[dev], dev_qd_W_10[dev], dev, NDev);
  }
}

#ifdef __cplusplus
}
#endif

