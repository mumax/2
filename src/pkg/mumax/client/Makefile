include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/client

GOFILES=\
	client.go\
	main.go\
	ipc.go\
	interpreter.go\
	apigen.go\
	apigen_lang.go\
	apigen_java.go\
	apigen_python.go\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../../cuda/driver/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
