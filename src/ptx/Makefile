include ../Make.inc

ptxfiles=\
	memory.ptx\
	math.ptx\
	debug.ptx\

all: $(ptxfiles)

$(ptxfiles): %.ptx: %.cu
	rm -f $@
	 $(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

.PHONY: install
install:

.PHONY: test
test: all

.PHONY: bench
bench:

