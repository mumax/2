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
	index.go\
	sizes.go\
	size.go\
	ansi.go\
	constants.go\
	timer.go\

CGOFILES=\

#GC+=\
#LD+=\
#CGO_CFLAGS+=-I/usr/local/cuda/include/
#UNAME := $(shell uname)
#ifeq ($(UNAME), Darwin) # Os X
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib -lcudart
#else # Linux and other Unix based systems
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
#endif

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
