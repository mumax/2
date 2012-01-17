# general options to be included in all Makefiles
# options for optimized build

# flags for the nvcc compiler
NVCC=nvcc 
NVCC+=\
  --compiler-options -Wall\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --use_fast_math\
  
NVCCWERRORLESS=nvcc
NVCCWERRORLESS+=\
  --compiler-options -Wall\
  --compiler-options -fPIC\
  --use_fast_math\

# flags for the gcc compiler
# As CUDA is delivered only in 32 bits on Os X, architecture is forced to i386 on Darwin platform

CC=g++\

CC+=\
 -I/usr/local/cuda/include\
 -Wall\
 -O3\

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin) # Os X
CC+=\
 -arch i386
export GOARCH=386

else # Linux and other Unix based systems

CC+=\

endif
