include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=cuda/cufft

CGOFILES=\
	plan.go\
	type.go\
	result.go\
	mode.go\

GOFILES=\
	doc.go\

CGO_CFLAGS+=-I/usr/local/cuda/include/
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia-current/ -lcuda -lcudart -lcufft

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
