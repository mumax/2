/**
  * @file
  * This file implements the torque according to Landau-Lifshitz.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _Q_H_
#define _Q_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// calculates heat densities for various subsystems
DLLEXPORT void Q_async(float** Qi,
                         float** Ti, float** Tj, float** Tk, 
                         float** Q, 
                         float** gamma_i, 
                         float** Gij, float** Gik,
                         float** k,
                         float QMul, 
                         float gamma_iMul, 
                         float GijMul, float GikMul, float kMul,
                         int isCofT,
                         const int sx, const int sy, const int sz,
                         const float csx, const float csy, const float csz,
                         const int pbc_x, const int pbc_y, const int pbc_z,
                         CUstream* stream);
                         
//DLLEXPORT void Qe_async(float** Qe,
//                         float** Te, float** Tl, float** Ts, 
//                         float** Q, 
//                         float** gamma_e, 
//                         float** Gel, float** Ges,
//                         float QMul, 
//                         float gamma_eMul, 
//                         float GelMul, float GesMul, 
//                         CUstream* stream, int Npart);
//                         
//DLLEXPORT void Qs_async(float** Qs,
//                         float** Te, float** Tl, float** Ts,
//                         float** Cs, 
//                         float** Gsl, float** Ges, 
//                         float CsMul, 
//                         float GslMul, float GesMul, CUstream* stream, int Npart);
//                         
//DLLEXPORT void Ql_async(float** Ql,
//                         float** Te, float** Tl, float** Ts,
//                         float** Cl, 
//                         float** Gel, float** Gsl,
//                         float ClMul,
//                         float GelMul, float GslMul, 
//                         CUstream* stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
