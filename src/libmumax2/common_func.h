/**
  * @file
  * This file implements common function typically required for calculus.
  *
  *
  * @author Mykola Dvornik
  */

#ifndef _COMMON_FUNC_H
#define _COMMON_FUNC_H

#include <cuda.h>

#include "stdio.h"
// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// mod
struct float5 {
    float x;
    float y;
    float z;
    float w;
    float t;
};

typedef struct float5 float5;

inline __host__ __device__ float5 make_float5(float x, float y, float z, float w, float t){
    float5 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.t = t;
    return a;
} 

inline __host__ __device__ int Mod(int a, int b){
	return (a%b+b)%b;
}


// dot product
inline __host__ __device__ float dotf(float3 a, float3 b)
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 crossf(float3 a, float3 b)
{ 
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// lenght of the 3-components vector
inline __host__ __device__ float len(float3 a){
	return sqrtf(dotf(a,a));
}

// normalize the 3-components vector
inline __host__ __device__ float3 normalize(float3 a){
    float veclen = (len(a) != 0.0f) ? 1.0f / len(a) : 0.0f;
	return make_float3(a.x * veclen, a.y * veclen, a.z * veclen);
}

#endif
