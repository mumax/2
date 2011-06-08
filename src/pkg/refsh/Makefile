include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=refsh

GOFILES=\
	caller.go\
	flags2.go\
	help.go\
	parse.go\
	readline.go\
	refsh.go\
	resolve.go\
	error.go\

CGOFILES=\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
