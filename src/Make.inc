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
# As CUDA is delivered only in 32 bits on Os X, architecture is forced to i386 on Darwin platform

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin) # Os X
CC=g++
CC+=\
 -Wall\
 -g\
 -arch i386
export GOARCH=386
else # Linux and other Unix based systems
CC=g++
CC+=\
 -Wall\
 -g
endif
