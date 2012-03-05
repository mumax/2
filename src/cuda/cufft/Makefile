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

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin) # Os X
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib -lcuda -lcudart -lcufft
else # Linux and other Unix based systems
CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia -L/usr/lib64/nvidia -L/usr/lib/nvidia-current -lcuda -lcudart -lcufft
endif

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
