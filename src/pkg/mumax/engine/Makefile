include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/engine

GOFILES=\
	apigen_c.go\
	apigen.go\
	apigen_java.go\
	apigen_lang.go\
	apigen_lua.go\
	apigen_python.go\
	client.go\
	engine.go\
	exec.go\
	field.go\
	field_test.go\
	gob_test.go\
	init_test.go\
	interpreter.go\
	io.go\
	ipc_test.go\
	main.go\
	rpc.go\
	rpc_test.go\
	run.go\
	units.go\
	universe.go\
	universe_test.go\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
