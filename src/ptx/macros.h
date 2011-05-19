
#ifndef gpu_conf_h
#define gpu_conf_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro for 1D index "i" in a CUDA kernel.
 @code
  i = threadindex;
 @endcode
 */
#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/**
 * Macro for integer division, but rounded UP
 */
#define divUp(x, y) ( (((x)-1)/(y)) +1 )
// It's almost like LISP ;-)

               

#ifdef __cplusplus
}
#endif

#endif
