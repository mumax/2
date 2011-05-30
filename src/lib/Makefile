ofiles=\
	reduce.o\

NVCC=nvcc 

all: $(ofiles)

$(ofiles): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

.PHONY: clean
clean:
	rm -f *.o

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

