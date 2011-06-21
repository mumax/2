include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/engine

GOFILES=\
	engine.go\
	universe.go\
	units.go\
	field.go\
	main.go\
	rpc.go\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
