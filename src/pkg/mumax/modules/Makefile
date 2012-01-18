include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/modules

GOFILES=\
	mod_micromag.go\
	magnetization.go\
	update_norm.go\
	hfield.go\
	mod_zeeman.go\
	mod_exchange6.go\
	mod_efield.go\
	mod_coulomb.go\
	mod_uniaxialanisotropy.go\
	mod_llg.go\
	mod_demagexchange.go\
	kernel.go\
	kernel_electrostatic.go\
	kernel_magnetostatic.go\
	kernel_biotsavart.go\
	kernel_exchange.go\
	vector.go\

CGOFILES=\


CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
