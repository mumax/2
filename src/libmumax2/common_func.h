/**
  * @file
  * This file implements common function typically required for calculus.
  *
  *
  * @author Mykola Dvornik
  */

#ifndef _COMMON_FUNC_H
#define _COMMON_FUNC_H

// mod
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
