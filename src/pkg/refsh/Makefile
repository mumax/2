include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/refsh

GOFILES=\
	caller.go\
	flags2.go\
	help.go\
	parse.go\
	readline.go\
	refsh.go\
	resolve.go\

CGOFILES=\

GC+=\
	-I ../common/_obj\

#LD+=\
#CGO_CFLAGS+=-I/usr/local/cuda/include/
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
