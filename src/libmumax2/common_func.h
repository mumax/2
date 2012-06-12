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

typedef double real;

struct float5 {
    float x;
    float y;
    float z;    
    float w;
    float v;
};

typedef struct float5 float5;

struct int6 {
    int x;
    int y;
    int z;    
    int w;
    int v;
    int t;
};

typedef struct int6 int6;

struct real5 {
    real x;
    real y;
    real z;
    real w;
    real v;
};

typedef struct real5 real5;

struct real6 {
    real x;
    real y;
    real z;
    real w;
    real v;
    real t;
};

typedef struct real6 real6;

struct real7 {
    real x;
    real y;
    real z;
    real w;
    real v;
    real t;
    real q;
};

typedef struct real7 real7;

struct real4 {
    real x;
    real y;
    real z;
    real w;
};

typedef struct real4 real4;

struct real3 {
    real x;
    real y;
    real z;
};

typedef struct real3 real3;

inline __host__ __device__ int6 make_int6(int x, int y, int z, int w, int v, int t){
    int6 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    a.t = t;
    return a;
} 

inline __host__ __device__ float5 make_float5(float x, float y, float z, float w, float v){
    float5 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    return a;
} 

inline __host__ __device__ real7 make_real7(real x, real y, real z, real w, real t, real q){
    real7 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.t = t;
    a.q = q;
    return a;
}

inline __host__ __device__ real5 make_real5(real x, real y, real z, real w, real v){
    real5 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    return a;
}

inline __host__ __device__ real6 make_real6(real x, real y, real z, real w, real v, real t){
    real6 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    a.t = t;
    return a;
}

inline __host__ __device__ real3 make_real3(real x, real y, real z){
    real3 a;
    a.x = x;
    a.y = y;
    a.z = z;
    return a;
}

// Python-like modulus
inline __host__ __device__ int Mod(int a, int b){
	return (a%b+b)%b;
}

// dot product
inline __host__ __device__ float dotf(float3 a, float3 b)
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(real3 a, real3 b)
{ 
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 crossf(float3 a, float3 b)
{ 
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ real3 cross(real3 a, real3 b)
{ 
	return make_real3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// lenght of the 3-components vector
inline __host__ __device__ float len(float3 a){
	return sqrtf(dotf(a,a));
}

inline __host__ __device__ real len(real3 a){
	return sqrt(dot(a,a));
}

// normalize the 3-components vector
inline __host__ __device__ float3 normalize(float3 a){
    float veclen = (len(a) != 0.0f) ? 1.0f / len(a) : 0.0f;
	return make_float3(a.x * veclen, a.y * veclen, a.z * veclen);
}

inline __host__ __device__ real3 normalize(real3 a){
    real veclen = (len(a) != 0.0) ? 1.0 / len(a) : 0.0;
	return make_real3(a.x * veclen, a.y * veclen, a.z * veclen);
}

#endif
