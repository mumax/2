include $(GOROOT)/src/Make.inc

executables=\
	mumax2\
	apigen\
	#mumax2-engine\

GC+=\
	-I../common/_obj\
	-I../client/_obj\
	-I../engine/_obj\
	-I../apigen/_obj\
	
LD+=\
	-L../common/_obj\
	-L../client/_obj\
	-L../engine/_obj\
	-L../apigen/_obj\

all: $(executables)

$(executables):	%: %.go
	$(GC) -I_obj $@.go
	$(LD) -L_obj -o $@ $@.$O

.PHONY:	clean
clean:
	rm -f  $(executables) *.a *.$O $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test:
	
.PHONY: bench
bench:

