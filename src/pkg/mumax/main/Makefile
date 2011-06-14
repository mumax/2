include $(GOROOT)/src/Make.inc

executables=\
	mumax2-client\
	#apigen\
	#mumax2-engine\

GC+=\
	-I../common/_obj\
	-I../client/_obj\
	-I../engine/_obj\
	-I../apigen/_obj\
	-I../../cuda/driver/_obj\
	
LD+=\
	-L../common/_obj\
	-L../client/_obj\
	-L../engine/_obj\
	-L../apigen/_obj\
	-L../../cuda/driver/_obj\

all: $(executables)

# always build executables
# build is fast and dependencies complex
.PHONY: $(executables)
$(executables):	%:
	$(GC) -I_obj $@.go
	$(LD) -L_obj -o $@ $@.$O
	cp -f $@ ../../../../bin

.PHONY:	clean
clean:
	rm -f  $(executables) *.a *.$O $(CLEANFILES) mumax2.py mumax2.java mumax2.h mumax2.lua

.PHONY: install
install:

.PHONY: test
test: 

	
.PHONY: bench
bench:

