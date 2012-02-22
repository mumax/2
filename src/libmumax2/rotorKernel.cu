#include "rotorKernel.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif


/// @author Ben Van de Wiele

#define BLOCKSIZE 16 ///@todo use device properties

__device__ float getRotorKernelElement(int N0, int N1, int N2, int comp, int a, int b, int c, int per0, int per1, int per2, 
                                  float cellX, float cellY, float cellZ, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  
  int cutoff = 10000;    //square of the cutoff where the interaction is computed for dipole in the center in stead of magnetized volume

  
  // for elements in Kernel component gxx _________________________________________________________
    if (comp==0){
      result = 0.0f;
    }
  // ______________________________________________________________________________________________

  // for elements in Kernel component gyy _________________________________________________________
    if (comp==1){
      result = 0.0f;
    }
  // ______________________________________________________________________________________________

  // for elements in Kernel component gyy _________________________________________________________
    if (comp==2){
      result = 0.0f;
    }
  // ______________________________________________________________________________________________

  // for elements in Kernel component gyz and gzy _________________________________________________
    if (comp==3 || comp==6){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i*cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellY + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellZ + dev_qd_P_10_Z[cnt3];
                result += cellX * cellY * cellZ / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( x*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ *
                    ( (i*cellX) * __powf(r2,-1.5f) );
        }
        
        if (r2_int==0)
          result = 0.0;
        
      }
      if (comp==3)
        result *= 1.0f/4.0f/3.14159265f;
      if (comp==6)
        result *= -1.0f/4.0f/3.14159265f;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxz and gzx _________________________________________________
    if (comp==4 || comp==7){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i*cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellY + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellZ + dev_qd_P_10_Z[cnt3];
                result += cellX * cellY * cellZ / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( y*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ *
                    ( (j*cellX) * __powf(r2,-1.5f) );
        }
        
        if (r2_int==0)
          result = 0.0;
        
      }
      if (comp==4)
        result *= -1.0f/4.0f/3.14159265f;
      if (comp==7)
        result *= 1.0f/4.0f/3.14159265f;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxy and gyx _________________________________________________
    if (comp==5 || comp==8){
      for(int cnta=-per0; cnta<=per0; cnta++)
      for(int cntb=-per1; cntb<=per1; cntb++)
      for(int cntc=-per2; cntc<=per2; cntc++){

        int i = a + cnta*N0/2;
        int j = b + cntb*N1/2;
        int k = c + cntc*N2/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<cutoff){
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i*cellX + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellY + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellZ + dev_qd_P_10_Z[cnt3];
                result += cellX * cellY * cellZ / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( z*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*cellX)*(i*cellX) + (j*cellY)*(j*cellY) + (k*cellZ)*(k*cellZ);
          result += cellX * cellY * cellZ *
                    ( (k*cellX) * __powf(r2,-1.5f) );
        }
        
        if (r2_int==0)
          result = 0.0;

       
      }
      if (comp==5)
        result *= 1.0f/4.0f/3.14159265f;
      if (comp==8)
        result *= -1.0f/4.0f/3.14159265f;

    }
  // ______________________________________________________________________________________________



  return( result );
}





__global__ void initRotorKernelElementKern (float *data, int comp, 
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
            getRotorKernelElement(N0, N1, N2, comp, i, j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0)
          data[(N0-i)*N12 + j*N2 + k] = 
            getRotorKernelElement(N0, N1, N2, comp, -i, j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (k>0)
          data[i*N12 + j*N2 + N2-k] = 
            getRotorKernelElement(N0, N1, N2, comp, i, j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0 && k>0)
          data[(N0-i)*N12 + j*N2 + N2-k] = 
            getRotorKernelElement(N0, N1, N2, comp, -i, j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
      }
      if (j2>N1/2){
          data[i*N12 + j*N2 + k] = 
            getRotorKernelElement(N0, N1, N2, comp, i, -N1+j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0)
          data[(N0-i)*N12 + j*N2 + k] = 
            getRotorKernelElement(N0, N1, N2, comp, -i, -N1+j2, k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (k>0)
          data[i*N12 + j*N2 + N2-k] = 
            getRotorKernelElement(N0, N1, N2, comp, i, -N1+j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
        if (i>0 && k>0)
          data[(N0-i)*N12 + j*N2 + N2-k] = 
            getRotorKernelElement(N0, N1, N2, comp, -i, -N1+j2, -k, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10, dev_qd_W_10);
      }
    }
    
  }
  
  return;
}



void initRotorKernelElementAsync(float **data, int comp,                    /// data array and component
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
    initRotorKernelElementKern <<<gridSize, blockSize, 0, cudaStream_t(streams[dev])>>> 
      (data[dev], comp, N0, N1, N2, N1part, per0, per1, per2, cellX, cellY, cellZ, dev_qd_P_10[dev], dev_qd_W_10[dev], dev, NDev);
  }
}




#ifdef __cplusplus
}
#endif

