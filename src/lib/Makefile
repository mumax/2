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
	fft.o\
	exchange6.o\

OMFSOURCES=OMFHeader.cpp OMFImport.cpp
OMFSTATICLIBS=$(OMFSOURCES:.cpp=.a)

all: libmultigpu.so omflibs

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmultigpu.so: $(OFILES) libmultigpu.h
	$(NVCC) -shared $(OFILES) -o libmultigpu.so
	cp -f $(CURDIR)/libmultigpu.so $(CURDIR)/../../lib
	ln -sf $(CURDIR)/libmultigpu.so ../pkg/mumax/gpu

omflibs: $(OMFSTATICLIBS)
	cp $(CURDIR)/libOMFImport.a $(CURDIR)/../../lib
	cp $(CURDIR)/libOMFHeader.a $(CURDIR)/../../lib

%.a: %.o
	$(CC) $(CFLAGS) $(<:.o=.cpp) -o $(@:.a=.o)
	ar -cvq lib$@ $<

.PHONY: clean
clean:
	rm -f *.o *.so

install: all

.PHONY: test
test:

.PHONY: bench
bench:

