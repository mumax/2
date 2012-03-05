include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/ovf

GOFILES=\
	common.go\
	doc.go\
	omf_input.go\
	omf_output.go\
	ovf2_output.go\

CGOFILES=\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
