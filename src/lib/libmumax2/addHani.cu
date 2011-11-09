#include "addHani.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void addHaniUniaxialKern (float *hx, float *hy, float *hz, 
                                     float *mx, float *my, float *mz,
                                     float *Ku_map, float Ku_mul, 
                                     float *anisU_mapx, float anisU_mulx,
                                     float *anisU_mapy, float anisU_muly,
                                     float *anisU_mapz, float anisU_mulz,
                                     int Npart){

  int i = threadindex;
  if (i < Npart){
    float Ku;
    if (Ku_map==NULL)
      Ku = Ku_mul;
    else
      Ku = Ku_mul*Ku_map[i];

    float ux, uy, uz;
    if (anisU_mapx ==NULL){
      ux = anisU_mulx;
      uy = anisU_muly;
      uz = anisU_mulz;
    }
    else{
      ux = anisU_mulx*anisU_mapx[i];
      uy = anisU_muly*anisU_mapy[i];
      uz = anisU_mulz*anisU_mapz[i];
    }
    
    float mu = mx[i]*ux + my[i]*uy + mz[i]*uz;
    hx[i] += Ku*mu*ux;
    hy[i] += Ku*mu*ux;
    hz[i] += Ku*mu*ux;
  }

}


void addHaniUniaxialAsync(float **hx, float **hy, float **hz, 
                          float **mx, float **my, float **mz,
                          float **Ku_map, float Ku_mul, 
                          float **anisU_mapx, float anisU_mulx,
                          float **anisU_mapy, float anisU_muly,
                          float **anisU_mapz, float anisU_mulz,
                          CUstream* stream, int Npart
                          ){
  dim3 gridSize, blockSize;
  make1dconf(Npart, &gridSize, &blockSize);

<<<<<<< HEAD
  for (int i=0; i<nDevice(); i++){
    assert(hx[i] != NULL);
    assert(hy[i] != NULL);
    assert(hz[i] != NULL);
=======
  for (int i=0, i<nDevice(); i++){
    assert(Hx[i] != NULL);
    assert(Hy[i] != NULL);
    assert(Hz[i] != NULL);
>>>>>>> anisotropy files added, not yet compiled
    assert(mx[i] != NULL);
    assert(my[i] != NULL);
    assert(mz[i] != NULL);
    gpu_safe(cudaSetDevice(deviceId(i)));
<<<<<<< HEAD
    addHaniUniaxialKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (hx[i],hy[i],hz[i],  
=======
    addHaniUniaxialKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (Hx[i],Hy[i],Hz[i],  
>>>>>>> anisotropy files added, not yet compiled
                                                                               mx[i],my[i],mz[i], 
                                                                               Ku_map[i], Ku_mul,
                                                                               anisU_mapx[i], anisU_mulx,
                                                                               anisU_mapy[i], anisU_muly,
                                                                               anisU_mapz[i], anisU_mulz,
                                                                               Npart);
  
  }
}



<<<<<<< HEAD
=======


>>>>>>> anisotropy files added, not yet compiled
__global__ void addHaniCubicKern (float *hx, float *hy, float *hz, 
                                  float *mx, float *my, float *mz,
                                  float *K1_map, float K1_mul, 
                                  float *K2_map, float K2_mul, 
                                  float *anisU1_mapx, float anisU1_mulx,
                                  float *anisU1_mapy, float anisU1_muly,
                                  float *anisU1_mapz, float anisU1_mulz,
                                  float *anisU2_mapx, float anisU2_mulx,
                                  float *anisU2_mapy, float anisU2_muly,
                                  float *anisU2_mapz, float anisU2_mulz,
                                  int Npart){

  int i = threadindex;
  if (i < Npart){
    float K1;
    if (K1_map==NULL)
      K1 = K1_mul;
    else
      K1 = K1_mul*K1_map[i];

    float K2;
    if (K1_map==NULL)
      K2 = K2_mul;
    else
      K2 = K2_mul*K2_map[i];
    
    float u1x, u1y, u1z, u2x, u2y, u2z;
    if (anisU1_mapx ==NULL){
      u1x = anisU1_mulx;
      u1y = anisU1_muly;
      u1z = anisU1_mulz;
      u2x = anisU2_mulx;
      u2y = anisU2_muly;
      u2z = anisU2_mulz;
    }
    else{
      u1x = anisU1_mulx*anisU1_mapx[i];
      u1y = anisU1_muly*anisU1_mapy[i];
      u1z = anisU1_mulz*anisU1_mapz[i];
      u2x = anisU2_mulx*anisU2_mapx[i];
      u2y = anisU2_muly*anisU2_mapy[i];
      u2z = anisU2_mulz*anisU2_mapz[i];
    }
    
      // computation third anisotropy axis
    float u3x = u1y*u2z - u1z*u2y;
    float u3y = u1x*u2z - u1z*u2x;
    float u3z = u1x*u2y - u1y*u2x;
    
      // projections of m on anisotropy axes
    float a0 = mx[i]*u1x + my[i]*u1y + mz[i]*u1z;
    float a1 = mx[i]*u2x + my[i]*u2y + mz[i]*u2z;
    float a2 = mx[i]*u3x + my[i]*u3y + mz[i]*u3z;
    
      // squared
    float a00 = a0*a0;
    float a11 = a1*a1;
    float a22 = a2*a2;

      // differentiated energy expressions
    float dphi_0 = K1 * (a11+a22) * a0  +  K2 * a0  *a11 * a22;
    float dphi_1 = K1 * (a00+a22) * a1  +  K2 * a00 *a1  * a22;
    float dphi_2 = K1 * (a00+a11) * a2  +  K2 * a00 *a11 * a2 ;
    
      // adding hani to heff
    hx[i] += - dphi_0*u1x - dphi_1*u2x - dphi_2*u3x;
    hy[i] += - dphi_0*u1y - dphi_1*u2y - dphi_2*u3y;
    hz[i] += - dphi_0*u1z - dphi_1*u2z - dphi_2*u3z;

  }

}



void addHaniCubicAsync(float **hx, float **hy, float **hz, 
                       float **mx, float **my, float **mz,
                       float **K1_map, float K1_mul, 
                       float **K2_map, float K2_mul, 
                       float **anisU1_mapx, float anisU1_mulx,
                       float **anisU1_mapy, float anisU1_muly,
                       float **anisU1_mapz, float anisU1_mulz,
                       float **anisU2_mapx, float anisU2_mulx,
                       float **anisU2_mapy, float anisU2_muly,
                       float **anisU2_mapz, float anisU2_mulz,
                       CUstream* stream, int Npart
                       ){
  dim3 gridSize, blockSize;
  make1dconf(Npart, &gridSize, &blockSize);

<<<<<<< HEAD
  for (int i=0; i<nDevice(); i++){
    assert(hx[i] != NULL);
    assert(hy[i] != NULL);
    assert(hz[i] != NULL);
=======
  for (int i=0, i<nDevice(); i++){
    assert(Hx[i] != NULL);
    assert(Hy[i] != NULL);
    assert(Hz[i] != NULL);
>>>>>>> anisotropy files added, not yet compiled
    assert(mx[i] != NULL);
    assert(my[i] != NULL);
    assert(mz[i] != NULL);
    gpu_safe(cudaSetDevice(deviceId(i)));
<<<<<<< HEAD
    addHaniCubicKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (hx[i],hy[i],hz[i],  
                                                                            mx[i],my[i],mz[i], 
                                                                            K1_map[i], K1_mul,
                                                                            K2_map[i], K2_mul,
                                                                            anisU1_mapx[i], anisU1_mulx,
                                                                            anisU1_mapy[i], anisU1_muly,
                                                                            anisU1_mapz[i], anisU1_mulz,
                                                                            anisU2_mapx[i], anisU2_mulx,
                                                                            anisU2_mapy[i], anisU2_muly,
                                                                            anisU2_mapz[i], anisU2_mulz,
                                                                            Npart);
=======
    addHaniCubicKern <<<gridSize, blockSize, 0, cudaStream_t(stream[i])>>> (Hx[i],Hy[i],Hz[i],  
                                                                               mx[i],my[i],mz[i], 
                                                                               K1_map[i], K1_mul,
                                                                               K2_map[i], K2_mul,
                                                                               anisU1_mapx[i], anisU1_mulx,
                                                                               anisU1_mapy[i], anisU1_muly,
                                                                               anisU1_mapz[i], anisU1_mulz,
                                                                               anisU2_mapx[i], anisU2_mulx,
                                                                               anisU2_mapy[i], anisU2_muly,
                                                                               anisU2_mapz[i], anisU2_mulz,
                                                                               Npart);
>>>>>>> anisotropy files added, not yet compiled
  
  }
}





<<<<<<< HEAD
=======


>>>>>>> anisotropy files added, not yet compiled
#ifdef __cplusplus
}
#endif
