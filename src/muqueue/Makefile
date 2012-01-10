include $(GOROOT)/src/Make.inc

muqueue: *.go
	$(GC) -I_obj *.go 
	$(LD) -L_obj -o $@ *.$O


.PHONY:	clean
clean:
	rm -f  muqueue *.a *.$O $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test: 

.PHONY: bench
bench:

