include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=module/slonczewski_torque

GOFILES=\
	doc.go\
	slonczewski.go\

CGOFILES=\
	wrapper.go\


CGO_CFLAGS+=\
	-I/usr/local/cuda/include\
	-I../../libmumax2\

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin) # OS X
CGO_LDFLAGS+=\
	-L/usr/local/cuda/lib\
   	-L../../../libmumax2
else # Linux and other Unix based systems
CGO_LDFLAGS+=\
	-L/usr/local/cuda/lib\
   	-L/usr/local/cuda/lib64\
	-L/usr/lib/nvidia-current\
	-L/usr/lib/nvidia\
	-L/usr/lib64/nvidia\
   	-L../../libmumax2
endif


CGO_LDFLAGS+=\
	-lcuda\
   	-lcudart\
	libmumax2.so\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
