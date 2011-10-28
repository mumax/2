include ../Make.inc

OFILES=\
	gpu_safe.o\
	gpu_conf.o\
	gpu_properties.o\
	multigpu.o\
	reduce.o\
	add.o\
	torque.o\
	normalize.o\
	pad.o\
	transpose.o\
	exchange6.o\

all: libmultigpu.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmultigpu.so: $(OFILES) libmultigpu.h
	echo " *** USING DEBUG FLAGS *** "
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

