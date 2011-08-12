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
	#safe_launch.go\
	#safe_memory.go\

GOFILES=\
	array.go\
	doc.go\

CGO_CFLAGS+=-I/usr/local/cuda/include/
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
