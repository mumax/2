include $(GOROOT)/src/Make.inc

executables=\
	mumax2\
	apigen\

GC+=\
	-I../pkg/mumax/common/_obj\
	-I../pkg/mumax/host/_obj\
	-I../pkg/mumax/gpu/_obj\
	-I../pkg/mumax/engine/_obj\
	-I../pkg/mumax/frontend/_obj\
	-I../pkg/mumax/apigen/_obj\
	-I../pkg/mumax/queue/_obj\
	-I../pkg/mumax/../cuda/driver/_obj\
	-I../pkg/mumax/../cuda/cufft/_obj\
	-I../pkg/mumax/../cuda/runtime/_obj\
	
LD+=\
	-L../pkg/mumax/common/_obj\
	-L../pkg/mumax/host/_obj\
	-L../pkg/mumax/gpu/_obj\
	-L../pkg/mumax/engine/_obj\
	-L../pkg/mumax/frontend/_obj\
	-L../pkg/mumax/apigen/_obj\
	-L../pkg/mumax/queue/_obj\
	-L../pkg/mumax/../cuda/driver/_obj\
	-L../pkg/mumax/../cuda/cufft/_obj\
	-L../pkg/mumax/../cuda/runtime/_obj\

all: $(executables) 

# always build executables
# build is fast and dependencies complex
.PHONY: $(executables)
$(executables):	%:
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

