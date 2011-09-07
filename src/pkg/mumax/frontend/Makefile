include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/frontend

GOFILES=\
	main.go\
	client.go\
	client_main.go\
	test_main.go\
	jsonrpc.go\
	io.go\
	addmethods.go\
	version.go\
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
