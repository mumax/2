include ../Make.inc

OFILES=\
	gpu_safe.o\
	gpu_conf.o\
	gpu_properties.o\
	add.o\
	reduce.o\
	multigpu.o\

all: libmultigpu.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmultigpu.so: $(OFILES) libmultigpu.h
	$(NVCC) -shared $(OFILES) -o libmultigpu.so
	cp -f $(CURDIR)/libmultigpu.so $(CURDIR)/../../lib
	ln -sf $(CURDIR)/libmultigpu.so ../pkg/mumax/gpu


.PHONY: clean
clean:
	rm -f *.o *.so

install: all

.PHONY: test
test:

.PHONY: bench
bench:

