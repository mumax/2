include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/engine

GOFILES=\
	engine.go\
	engine_graphviz.go\
	engine_step.go\
	api.go\
	mod_micromag.go\
	mod_micromagenergy.go\
	mod_spintorque.go\
	mod_testmod.go\
	quant.go\
	quantkind.go\
	torque_node.go\
	sum_node.go\
	updater.go\
	nop_updater.go\
	pointwise_updater.go\
	input.go\
	output.go\
	resample.go\
	testapi.go\

CGOFILES=\

GC+=\
	-I../../cuda/driver/_obj\
	-I../common/_obj\
	-I../host/_obj\
	-I../gpu/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
