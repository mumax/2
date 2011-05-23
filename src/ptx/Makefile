ptxfiles=\
	memory.ptx\

all: $(ptxfiles)

$(ptxfiles):
	 nvcc -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

.PHONY: install
test:

.PHONY: test
test:

.PHONY: bench
test:

