ptxfiles=\
	memory.ptx\

all: $(ptxfiles)

$(ptxfiles):
	 nvcc -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

.PHONY: test
test:

.PHONY: bench
test:

