include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/gpu

GOFILES=\
	doc.go\
	gpus.go\
	array.go\
	slice.go\
	module.go\
	module_load.go\
	module_ptxparse.go\
	module_find.go\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../host/_obj\

#LD+=\

#CGO_CFLAGS+=-I/usr/local/cuda/include/
#CGO_LDFLAGS+=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
