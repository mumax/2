include $(GOROOT)/src/Make.inc

executables=\
	mumax2\
	apigen\

all: $(executables) 

modules=$(wildcard ../*.mod)

# always build executables
# build is fast and dependencies complex
.PHONY: $(executables)

#$(GC) -I_obj $@.go $(modules)/pkg/import.go
mumax2:
	$(GC) -I_obj $@.go 
	$(LD) -L_obj -o $@ $@.$O

apigen:
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

