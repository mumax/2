include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/engine

GOFILES=\
	client.go\
	engine.go\
	apigen.go\
	apigen_lang.go\
	apigen_c.go\
	apigen_java.go\
	apigen_lua.go\
	apigen_python.go\
	exec.go\
	field.go\
	interpreter.go\
	io.go\
	main.go\
	server.go\
	units.go\
	universe.go\

CGOFILES=\

GC+=\
	-I../../cuda/driver/_obj\
	-I../common/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
