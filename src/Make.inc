# general options to be included in all Makefiles

# flags for the nvcc compiler
NVCC=nvcc 
NVCC+=\
  --compiler-options -Wall\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
#--use_fast_math\

# flags for the gcc compiler
CC=g++
CC+=\
 -Wall\
 -g\
