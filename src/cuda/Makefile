export GOPATH=$(CURDIR)/../..
all:
	echo GOPATH=$(GOPATH)
	go build cuda/driver
	go build cuda/runtime
	go build cuda/cufft
	go build cuda/curand

#dirs=\
#	driver\
#	runtime\
#	cufft\
#	curand\
#
#all:$(dirs)
#
#include ../GoDirs.pkg
