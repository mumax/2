include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/engine

GOFILES=\
	universe.go\
	units.go\
	field.go\
	ioformat.go\
	stephandler.go\
	server_api.go\
	server.go\
	client.go\
	rpc.go\
	rpc_refsh.go\
	main.go

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
