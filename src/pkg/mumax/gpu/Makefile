include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/gpu

GOFILES=\
	doc.go\
	gpus.go\
	array.go\
	stream.go\
	module.go\
	module_load.go\
	module_ptxparse.go\
	module_find.go\
	math.go\

CGOFILES=\
	cmath.go\

GC+=\
	-I../../cuda/driver/_obj\
	-I../../cuda/runtime/_obj\
	-I../common/_obj\
	-I../host/_obj\

CGO_CFLAGS+=\
	-I/usr/local/cuda/include/\
	-I../../../lib\

CGO_LDFLAGS+=\
	-L/usr/local/cuda/lib\
   	-L/usr/local/cuda/lib64\
	-L/usr/lib/nvidia-current/\
   	-L../../../../lib\

CGO_LDFLAGS+=\
	-lcuda\
   	-lcudart\
	libmumax2.so\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
