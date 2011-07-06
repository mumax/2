include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/frontend

GOFILES=\
	client.go\
	client_main.go\
	jsonrpc.go\
	io.go\
	main.go\
	addmethods.go\
	#interpreter.go\

CGOFILES=\

GC+=\
	-I../../cuda/driver/_obj\
	-I../common/_obj\
	-I../gpu/_obj\
	-I../host/_obj\
	-I../engine/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
