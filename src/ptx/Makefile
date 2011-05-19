ptxfiles=\
	memory.ptx\

all: $(ptxfiles)

memory.ptx: memory.cu
	nvcc -ptx memory.cu

.PHONY: clean
clean:
	rm -f *.ptx

