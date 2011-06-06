include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/client

GOFILES=\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../engine/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
