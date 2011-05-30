ptxfiles=\
	memory.ptx\
	debug.ptx\

NVCC=nvcc

all: $(ptxfiles)

$(ptxfiles): %.ptx: %.cu
	 $(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

