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
	module.go\
	mod_magnetization.go\
	mod_llg.go\
	mod_zeeman.go\
	mod_micromag.go\
	mod_hfield.go\
	mod_uniaxialanisotropy.go\
	quant.go\
	quantkind.go\
	unit.go\
	sum_updater.go\
	updater.go\
	nop_updater.go\
	pointwise_updater.go\
	normalize_updater.go\
	solver.go\
	euler_solver.go\
	resample.go\
	#output.go\
	#input.go\

CGOFILES=\

GC+=\
	-I../../cuda/driver/_obj\
	-I../common/_obj\
	-I../host/_obj\
	-I../gpu/_obj\
	-I../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
