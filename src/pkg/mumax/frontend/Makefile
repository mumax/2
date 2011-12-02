include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../libmumax2

TARG=mumax/frontend

GOFILES=\
	main.go\
	client.go\
	client_main.go\
	engine_main.go\
	test_main.go\
	jsonrpc.go\
	io.go\
	addmethods.go\
	version.go\
	#interpreter.go\

CGOFILES=\


CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
