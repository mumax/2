include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=cuda/runtime

CGOFILES=\
	error.go\
	memory.go\
	device.go\
	stream.go\
	thread.go\
	execution.go\
	peer.go\
	#safe_launch.go\
	#safe_memory.go\

GOFILES=\
	array.go\
	doc.go\

CGO_CFLAGS+=-I/usr/local/cuda/include/

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin) # Os X
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib -lcudart
else # Linux and other Unix based systems
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
endif

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
