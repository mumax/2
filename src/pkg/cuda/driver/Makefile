include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=cuda/driver

CGOFILES=\
	result.go\
	init.go\
	version.go\
	device.go\
	context.go\
	stream.go\
	memory.go\
	memset.go\
	module.go\
	function.go\
	execution.go\
	peer.go\


GOFILES=\
	closure.go\
	doc.go\

CGO_CFLAGS+=-I/usr/local/cuda/include/
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia-current/ -lcuda -lcudart 

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
