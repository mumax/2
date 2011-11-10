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
	copypad.o\
	transpose.o\
	exchange6.o\
# 	addHani.o\

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
	$(CC) $(CFLAGS) $(<:.o=.cpp) -o $(@:.a=.o) -I.
	ar -cvq lib$@ $<

.PHONY: clean
clean:
	rm -f *.o *.so

install: all

.PHONY: test
test:

.PHONY: bench
bench:

