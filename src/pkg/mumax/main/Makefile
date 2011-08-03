include $(GOROOT)/src/Make.inc

executables=\
	mumax2-bin\
	mumax2-apigen\
	mumax2-queue\

GC+=\
	-I../common/_obj\
	-I../host/_obj\
	-I../gpu/_obj\
	-I../engine/_obj\
	-I../frontend/_obj\
	-I../apigen/_obj\
	-I../queue/_obj\
	-I../../cuda/driver/_obj\
	
LD+=\
	-L../common/_obj\
	-L../host/_obj\
	-L../gpu/_obj\
	-L../engine/_obj\
	-L../frontend/_obj\
	-L../apigen/_obj\
	-L../queue/_obj\
	-L../../cuda/driver/_obj\

all: $(executables)

# always build executables
# build is fast and dependencies complex
.PHONY: $(executables)
$(executables):	%:
	$(GC) -I_obj $@.go
	$(LD) -L_obj -o $@ $@.$O
	mv -f $@ ../../../../bin/sub

.PHONY:	clean
clean:
	rm -f  $(executables) *.a *.$O $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test: 

.PHONY: bench
bench:

