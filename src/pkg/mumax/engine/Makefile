include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/engine

GOFILES=\
	engine.go\
	engine_graphviz.go\
	gridsize.go\
	api.go\
	module.go\
	texgen.go\
	mod_magnetization.go\
	mod_llg.go\
	mod_micromag.go\
	mod_hfield.go\
	mod_uniaxialanisotropy.go\
	mod_demagexchange.go\
	kernel.go\
	demag.go\
	exchange.go\
	mod_regions.go\
	quant.go\
	quantkind.go\
	unit.go\
	updater.go\
	update_pointwise.go\
	update_norm.go\
	update_sum.go\
	update_average.go\
	zeropad.go\
	verify.go\
	equation.go\
	solver.go\
	solve_euler.go\
	solve_heun.go\
	solve_rk12.go\
	resample.go\
	notifier.go\
	notify_autosave.go\
	notify_tabulate.go\
	outputformat.go\
	format_txt.go\
	format_binary.go\
	input.go\
	table.go\
	arraypool.go\
	#input.go\

CGOFILES=\


CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
