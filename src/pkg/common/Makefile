include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/common

GOFILES=\
	assert.go\
	error.go\
	math.go\
	strconv.go\
	misc.go\
	tensor/gpus.go\
	tensor/splice.go\
	tensor/vsplice.go\
	tensor/spliceio.go\
	tensor/tensor.go\
	doc.go\

CGOFILES=\

#CGO_CFLAGS+=-I/usr/local/cuda/include/
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
