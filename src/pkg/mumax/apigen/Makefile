include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

# to make tests run
LD_LIBRARY_PATH+=:../../../../lib

TARG=mumax/apigen

GOFILES=\
	apigen.go\
	apigen2.go\
	c.go\
	java.go\
	lang.go\
	lua.go\
	python.go \

CGOFILES=\

GC+=\
	-I../../cuda/driver/_obj\
	-I../common/_obj\
	-I../engine/_obj\
	-I../frontend/_obj\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so mumax2.py mumax2.pyc Mumax2.java Mumax2.class mumax2.lua
	
include $(GOROOT)/src/Make.pkg
