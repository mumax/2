include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/queue

GOFILES=\
	gpu.go\
	main.go\
	node.go\
	task.go\



CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so
	
include $(GOROOT)/src/Make.pkg
