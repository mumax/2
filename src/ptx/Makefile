ptxfiles=\
	memory.ptx\

all: $(ptxfiles)

$(ptxfiles):
	 nvcc -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

