# general options to be included in all Makefiles
# options for DEBUG

# flags for the nvcc compiler
NVCC=nvcc
NVCC+=\
  --compiler-options -Wall\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
  
NVCCWERRORLESS=nvcc
NVCCWERRORLESS+=\
  --compiler-options -Wall\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\

# flags for the gcc compiler
# As CUDA is delivered only in 32 bits on Os X, architecture is forced to i386 on Darwin platform

CC=g++\

CC+=\
 -I/usr/local/cuda/include\
 -Wall\
 -g\

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin) # Os X
CC+=\
 -arch i386
export GOARCH=386

else # Linux and other Unix based systems

CC+=\

endif
