include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/engine

GOFILES=\
	engine.go\
	engine_graphviz.go\
	api.go\
	module.go\
	mod_magnetization.go\
	mod_llg.go\
	mod_zeeman.go\
	mod_micromag.go\
	mod_hfield.go\
	mod_uniaxialanisotropy.go\
	mod_aexchange.go\
	mod_demagexchange.go\
	mod_exchange6.go\
	quant.go\
	quantkind.go\
	unit.go\
	updater.go\
	update_pointwise.go\
	update_norm.go\
	update_sum.go\
	zeropad.go\
	verify.go\
	solver.go\
	solve_euler.go\
	resample.go\
	notifier.go\
	notify_autosave.go\
	notify_tabulate.go\
	outputformat.go\
	format_txt.go\
	format_binary.go\
	format_omf.go\
	format_ovf2.go\
	table.go\
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
