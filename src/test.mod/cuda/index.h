/**
  * @file
  * Functions for debugging only.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _INDEX_H_
#define _INDEX_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif



/// @debug sets array[i,j,k] to its C-oder X (outer) index.
void setIndexX(float** dst, int N0, int N1, int N2);

/// @debug sets array[i,j,k] to its C-oder Y index.
void setIndexY(float** dst, int N0, int N1, int N2);

/// @debug sets array[i,j,k] to its C-oder Z (inner) index.
void setIndexZ(float** dst, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
