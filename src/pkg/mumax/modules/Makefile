include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/modules

GOFILES=\
	mod_zeeman.go\
	mod_exchange6.go\
	mod_current.go\
	kernel.go\

CGOFILES=\


CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
