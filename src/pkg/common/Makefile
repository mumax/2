include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/common

GOFILES=\
	doc.go\
	assert.go\
	error.go\
	logger.go\
	math.go\
	io.go\
	strconv.go\
	misc.go\
	gpus.go\
	array.go\
	array/splice.go\
	array/vsplice.go\
	array/spliceio.go\

CGOFILES=\

#GC+=\
#LD+=\
#CGO_CFLAGS+=-I/usr/local/cuda/include/
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
