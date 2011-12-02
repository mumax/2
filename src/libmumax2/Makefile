include ../Make.inc

CUFILES=\
	gpu_safe.cu\
	gpu_conf.cu\
	gpu_properties.cu\
	multigpu.cu\
	reduce.cu\
	add.cu\
	torque.cu\
	normalize.cu\
	copypad.cu\
	transpose.cu\
	exchange6.cu\
	kernelmul_micromag.cu\
 	uniform.cu\
	index.cu\

OFILES=$(CUFILES:.cu=.o)

all: libmumax2.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu

omflibs: $(OMFSTATICLIBS)
	cp $(CURDIR)/libOMFImport.a $(CURDIR)/../../lib
	cp $(CURDIR)/libOMFHeader.a $(CURDIR)/../../lib

%.a: %.o
	$(CC) $(<:.o=.cpp) -o $(@:.a=.o) -I.
	ar -cvq lib$@ $<

.PHONY: clean
clean:
	rm -f *.o *.so

install: all

.PHONY: test
test:

.PHONY: bench
bench:

