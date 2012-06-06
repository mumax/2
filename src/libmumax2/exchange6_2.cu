#include "exchange6_2.h"
#include "common_func.h"

#include "multigpu.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif


#define EXCH_BLOCK_X 16
#define EXCH_BLOCK_Y 8
#define I_OFF (EXCH_BLOCK_X+2)*(EXCH_BLOCK_Y+2)
#define J_OFF (EXCH_BLOCK_X+2)
///> important: for  6 neighbor EXCH_BLOCK_X*EXCH_BLOCK_Y needs to be larger than (or equal to) 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2)!! 
///> important: for 12 neighbor EXCH_BLOCK_X*EXCH_BLOCK_Y needs to be larger than (or equal to) 4*(EXCH_BLOCK_X + EXCH_BLOCK_Y)!! 

#define I_OFF2 (EXCH_BLOCK_X+4)*(EXCH_BLOCK_Y+4)
#define J_OFF2 (EXCH_BLOCK_X+4)


__global__ void exchange6_3DKern (float* h, float* m, float* mSat_map, float* Aex_map, float Aex2_Mu0Msat_mul, float* mPart0, float* mPart2,
                                  int N0, int N1Part, int N2,
                                  int periodic_X, int periodic_Y, int periodic_Z,
                                  float cellx_2, float celly_2, float cellz_2){

  float *hptr, result;
  int i, j, k, ind, ind_h, indg_in, indg_out, indg_h, active_in, active_out, mParth, mPartm;

  int Nyz = N1Part*N2; 
  int Nx_minus_1 = N0-1;
  int cnt = threadIdx.y*EXCH_BLOCK_X + threadIdx.x;
  
  float *mParts[3];
  mParts[0] = mPart0;
  mParts[1] = m;
  mParts[2] = mPart2;

  float mSat_mask = 1.0f;
  float Aex2_Mu0Msat = Aex2_Mu0Msat_mul / mSat_mask;    // 2 * Aex / Mu0 * Msat
  
  
// initialize shared memory ------------------------------------------------------------
  __shared__ float m_sh[3*I_OFF]; 

  
// initialize indices for halo elements ------------------------------------------------
  int halo = cnt < 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2);
  if (halo) {
    if (threadIdx.y<2) {                                // shared memory indices, y-halos (coalesced)
      j = threadIdx.y*(EXCH_BLOCK_Y+1) - 1;
      k = threadIdx.x;
    }
    else {                                              // shared memory indices, z-halos (not coalesced)
      j =  cnt/2 - EXCH_BLOCK_X - 1;
      k = (cnt%2)*(EXCH_BLOCK_X+1) - 1;
    }

    ind_h  = I_OFF + (j+1)*J_OFF + k+1 ;                // shared memory halo index
    m_sh[ind_h-I_OFF] = 0.0f;                           // initialize to zero
    m_sh[ind_h] = 0.0f;
    m_sh[ind_h+I_OFF] = 0.0f;

    j = blockIdx.y*EXCH_BLOCK_Y + j;
    k = blockIdx.x*EXCH_BLOCK_X + k;                    //global indices

    mParth = 1;
    if (k==-1) {k= (periodic_Z) ? k=N2-1 : k=0;   }     // adjust for PBCs, Neumann boundary conditions if no PBCs
    if (k==N2) {k= (periodic_Z) ? k=0    : k=N2-1;}
    if (j==-1){ 
      if (periodic_Y) { mParth=0; j=N1Part-1; }
      else { j=0; }
    }
    if (j==N1Part){ 
      if (periodic_Y) { mParth=2; j=0; }
      else { j=N1Part-1; }
    }

    indg_h = j*N2 + k;                                  // global halo index

    halo = (j>=0) && (j<N1Part) && (k>=0) && (k<N2);
  }
// -------------------------------------------------------------------------------------


// initialize indices for main block ---------------------------------------------------
  j    = threadIdx.y;                                   // shared memory indices
  k    = threadIdx.x;
  ind  = I_OFF + (j+1)*J_OFF + k+1;
  m_sh[ind-I_OFF] = 0.0f;                               // initialize to zero
  m_sh[ind] = 0.0f;
  m_sh[ind+I_OFF] = 0.0f;

  j = blockIdx.y*EXCH_BLOCK_Y + j;                      // global indices
  k = blockIdx.x*EXCH_BLOCK_X + k;
  indg_out = j*N2 + k;
  active_out = (j<N1Part) && (k<N2);

  mPartm = 1;
  if (k==N2) {k= (periodic_Z) ? k=0 : k=N2-1;}       // adjust for PBCs, Neumann boundary conditions if no PBCs
  if (j==N1Part){ 
    if (periodic_Y) { mPartm=2; j=0; }
    else { j=N1Part-1; }
  }

  indg_in = j*N2 + k;
  active_in = (j<N1Part) && (k<N2);
// -------------------------------------------------------------------------------------


// if periodic_X: read last yz-plane of array ------------------------------------------
  if (periodic_X){
    if (active_in) 
      m_sh[ind  ] = mParts[mPartm][indg_in + Nx_minus_1*Nyz];
    if (halo) 
      m_sh[ind_h] = mParts[mParth][indg_h + Nx_minus_1*Nyz];
  }
// -------------------------------------------------------------------------------------

// read first yz plane of array --------------------------------------------------------
  if (active_in) 
    m_sh[ind   + I_OFF] = mParts[mPartm][indg_in];
  if (halo) 
    m_sh[ind_h + I_OFF] = mParts[mParth][indg_h];
// -------------------------------------------------------------------------------------


// perform the actual exchange computations --------------------------------------------
  for (i=0; i<N0; i++) {

    if (active_in) {

      // define prefactor based on central cell (!!)
      if ( mSat_map!=NULL ){ 
        mSat_mask = mSat_map[indg_in];
        if (mSat_mask==0.0f) { mSat_mask=1.0f; }     // do not divide by zero
      }
      if ( mSat_map!=NULL || Aex_map!=NULL )
        Aex2_Mu0Msat = (Aex2_Mu0Msat_mul / mSat_mask) * Aex_map[indg_in];     // 2 * Aex / Mu0 * Msat
      
      // deal with the data planes
      hptr = h + indg_out;                            // increase indices
      indg_in += Nyz;
      indg_out += Nyz;
      m_sh[ind - I_OFF] = m_sh[ind];                  // shift the two existing planes
      m_sh[ind]         = m_sh[ind + I_OFF];
      if (i<Nx_minus_1)
        m_sh[ind + I_OFF] = mParts[mPartm][indg_in];   // read new plane
      else if (periodic_X!=0)
        m_sh[ind + I_OFF] = mParts[mPartm][indg_in - Nx_minus_1*Nyz];      // PBCs
      else
        m_sh[ind + I_OFF] = m_sh[ind];                // Neumann boundary conditions
    }

    if (halo) {
      indg_h = indg_h + Nyz;                          // increase index
      m_sh[ind_h - I_OFF] = m_sh[ind_h];              // shift planes
      m_sh[ind_h]         = m_sh[ind_h + I_OFF];
      if (i<Nx_minus_1)
        m_sh[ind_h + I_OFF] = mParts[mParth][indg_h];  // read new plane
      else if (periodic_X!=0)
        m_sh[ind_h + I_OFF] = mParts[mParth][indg_h - Nx_minus_1*Nyz];     //PBCs
      else
        m_sh[ind_h + I_OFF] = m_sh[ind_h];            // Neumann boundary conditions
    }
    __syncthreads();

    if (active_out){
      result  = Aex2_Mu0Msat * cellx_2 * ( ( m_sh[ind-I_OFF]-m_sh[ind] ) + ( m_sh[ind+I_OFF]-m_sh[ind] ) );
      result += Aex2_Mu0Msat * celly_2 * ( ( m_sh[ind-J_OFF]-m_sh[ind] ) + ( m_sh[ind+J_OFF]-m_sh[ind] ) );
      result += Aex2_Mu0Msat * cellz_2 * ( ( m_sh[ind-1]    -m_sh[ind] ) + ( m_sh[ind+1]    -m_sh[ind] ) );
      *hptr = result;       // Not adding
    }
    __syncthreads();

  }

  return;
}



__global__ void exchange6_2DKern (float* h, float* m, float* mSat_map, float* Aex_map, float Aex2_Mu0Msat_mul, float* mPart0, float* mPart2,
                                  int N1Part, int N2,
                                  int periodic_Y, int periodic_Z,
                                  float celly_2, float cellz_2){

  float result;
  int j, k, ind, ind_h, indg_in, indg_out, indg_h, active_in, active_out, mParth, mPartm;

  int cnt = threadIdx.y*EXCH_BLOCK_X + threadIdx.x;
  
  float *mParts[3];
  mParts[0] = mPart0;
  mParts[1] = m;
  mParts[2] = mPart2;

  float mSat_mask = 1.0f;
  float Aex2_Mu0Msat = Aex2_Mu0Msat_mul / mSat_mask;    // 2 * Aex / Mu0 * Msat
  
 
// initialize shared memory ------------------------------------------------------------
  __shared__ float m_sh[I_OFF]; 

  
// initialize indices for halo elements ------------------------------------------------
  int halo = cnt < 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2);
  if (halo) {
    if (threadIdx.y<2) {                                // shared memory indices, y-halos (coalesced)
      j = threadIdx.y*(EXCH_BLOCK_Y+1) - 1;
      k = threadIdx.x;
    }
    else {                                              // shared memory indices, z-halos (not coalesced)
      j =  cnt/2 - EXCH_BLOCK_X - 1;
      k = (cnt%2)*(EXCH_BLOCK_X+1) - 1;
    }

    ind_h  = (j+1)*J_OFF + k+1 ;                // shared memory halo index
    m_sh[ind_h] = 0.0f;                                 // initialize to zero

    j = blockIdx.y*EXCH_BLOCK_Y + j;
    k = blockIdx.x*EXCH_BLOCK_X + k;                    //global indices

    mParth = 1; // h refers to halo
    if (k==-1) {k= (periodic_Z) ? k=N2-1 : k=0;   }     // adjust for PBCs, Neumann boundary conditions if no PBCs
    if (k==N2) {k= (periodic_Z) ? k=0    : k=N2-1;}
    if (j==-1){ 
      if (periodic_Y) { mParth=0; j=N1Part-1; }
      else { j=0; }
    }
    if (j==N1Part){ 
      if (periodic_Y) { mParth=2; j=0; }
      else { j=N1Part-1; }
    }

    indg_h = j*N2 + k;                                  // global halo index

    halo = (j>=0) && (j<N1Part) && (k>=0) && (k<N2);
  }
// -------------------------------------------------------------------------------------


// initialize indices for main block ---------------------------------------------------
  j    = threadIdx.y;                                   // shared memory indices
  k    = threadIdx.x;
  ind  = (j+1)*J_OFF + k+1;
  m_sh[ind] = 0.0f;                                     // initialize to zero

  j = blockIdx.y*EXCH_BLOCK_Y + j;                      // global indices
  k = blockIdx.x*EXCH_BLOCK_X + k;
  indg_out = j*N2 + k;
  active_out = (j<N1Part) && (k<N2);

  mPartm = 1;   // m refers to Main block
  if (k==N2) {k= (periodic_Z) ? k=0 : k=N2-1;}          // adjust for PBCs, Neumann boundary conditions if no PBCs
  if (j==N1Part){ 
    if (periodic_Y) { mPartm=2; j=0; }
    else { j=N1Part-1; }
  }

  indg_in = j*N2 + k;
  active_in = (j<N1Part) && (k<N2);
// -------------------------------------------------------------------------------------


// read first yz plane of array --------------------------------------------------------
  if (active_in) 
    m_sh[ind  ] = mParts[mPartm][indg_in];
  if (halo) 
    m_sh[ind_h] = mParts[mParth][indg_h];
// -------------------------------------------------------------------------------------
  __syncthreads();


// perform the actual exchange computations --------------------------------------------
  if (active_out){

    // define prefactor based on central cell (!!)
    if ( mSat_map!=NULL ){ 
      mSat_mask = mSat_map[indg_in];
      if (mSat_mask==0.0f) { mSat_mask=1.0f; }     // do not divide by zero
    }
    if ( mSat_map!=NULL || Aex_map!=NULL )
      Aex2_Mu0Msat = (Aex2_Mu0Msat_mul / mSat_mask) * Aex_map[indg_in];     // 2 * Aex / Mu0 * Msat

    result  = Aex2_Mu0Msat * celly_2 * ( ( m_sh[ind-J_OFF]-m_sh[ind] ) + ( m_sh[ind+J_OFF]-m_sh[ind] ) );
    result += Aex2_Mu0Msat * cellz_2 * ( ( m_sh[ind-1]    -m_sh[ind] ) + ( m_sh[ind+1]    -m_sh[ind] ) );
    h[indg_out] = result;       // Not adding
  }
  __syncthreads();


  return;
}







// int mod(int a, int b){
// 	return (a%b+b)%b;
// }



#define BLOCKSIZE 16
__export__ void exchange6_2Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float** msat, float** aex, float Aex2_mu0MsatMul, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams){

  int bx = 1 + (N2-1)/EXCH_BLOCK_X;    // a grid has blockdim.z=1, so we use the x-component
  int by = 1 + (N1Part-1)/EXCH_BLOCK_Y;
  dim3 gridsize (bx, by);
  dim3 blocksize (EXCH_BLOCK_X, EXCH_BLOCK_Y);

  float cellx_2 = 1/(cellSizeX * cellSizeX);
  float celly_2 = 1/(cellSizeY * cellSizeY);
  float cellz_2 = 1/(cellSizeZ * cellSizeZ);

	int nDev = nDevice();

	float** H = hx;
  float** M = mx;
	for(int c=0; c<3; c++){        // for all 3 components

    if (c==0){H = hx; M = mx;}
		if (c==1){H = hy; M = my;}
		if (c==2){H = hz; M = mz;}

		for (int dev = 0; dev < nDev; dev++) {
      
			gpu_safe(cudaSetDevice(deviceId(dev)));
      
			// set up adjacent parts
			float* mPart0 = M[Mod(dev-1, nDev)];  // adjacent part for smaller Y reps. larger Y
			float* mPart2 = M[Mod(dev+1, nDev)];  // parts wrap around...
      if (N0>1)
        exchange6_3DKern<<<gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(H[dev], M[dev], msat[dev], aex[dev], Aex2_mu0MsatMul, mPart0, mPart2, N0, N1Part, N2, periodic0, periodic1, periodic2, cellx_2, celly_2, cellz_2);
      else
        exchange6_2DKern<<<gridsize, blocksize, 0, cudaStream_t(streams[dev])>>>(H[dev], M[dev], msat[dev], aex[dev], Aex2_mu0MsatMul, mPart0, mPart2, N1Part, N2, periodic1, periodic2, celly_2, cellz_2);
    }

  }
}


#ifdef __cplusplus
}
#endif

