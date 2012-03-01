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
	quant.go\
	quantkind.go\
	unit.go\
	updater.go\
	update_pointwise.go\
	update_sum.go\
	update_addterm.go\
	update_reduce.go\
	update_peak.go\
	sumnode.go\
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
	format_gplot.go\
	format_binary.go\
	format_png.go\
	input.go\
	table.go\
	arraypool.go\
	timederivative.go\
	format_vtk.go\
	#input.go\

CGOFILES=\
	lib_quant_init.go\
	

CGO_CFLAGS+=\
	-I/usr/local/cuda/include\
	-I../../../libmumax2\


UNAME := $(shell uname)

ifeq ($(UNAME), Darwin) # OS X
CGO_LDFLAGS+=\
	-L/usr/local/cuda/lib\
   	-L../../../libmumax2
else # Linux and other Unix based systems
CGO_LDFLAGS+=\
	-L/usr/local/cuda/lib\
   	-L/usr/local/cuda/lib64\
	-L/usr/lib/nvidia-current\
	-L/usr/lib/nvidia\
	-L/usr/lib64/nvidia\
   	-L../../../libmumax2
endif

CGO_LDFLAGS+=\
	-lcuda\
   	-lcudart\
   	-lmumax2\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
