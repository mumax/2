include $(GOROOT)/src/Make.inc

GOFILES=\
	mumax-engine.go\

GC+=\
	-I../common/_obj\
	-I../engine/_obj\
	
LD+=\
	-L../common/_obj\
	-L../engine/_obj\
	

main: $(GOFILES)
	$(GC) -I_obj $(GOFILES)
	$(LD) -L_obj -o $@ *.$O

.PHONY:	clean
clean:
	rm -f  main *.a *.$O $(CLEANFILES)
