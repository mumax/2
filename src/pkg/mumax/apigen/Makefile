include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/apigen

GOFILES=\
	main.go\
	lang.go\
	python.go\
	java.go\

CGOFILES=\

GC+=\
	-I../common/_obj\
	-I../client/_obj\
	-I../../refsh/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
