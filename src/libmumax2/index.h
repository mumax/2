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



void setIndexX(float** dst, int N0, int N1, int N2);
void setIndexY(float** dst, int N0, int N1, int N2);
void setIndexZ(float** dst, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
