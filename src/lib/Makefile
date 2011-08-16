include ../Make.inc

OFILES=\
	gpu_safe.o\
	gpu_conf.o\
	gpu_properties.o\
	add.o\
	reduce.o\
	multigpu.o\

all: libmumax2.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmumax2.so: $(OFILES) libmumax2.h
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	cp -f $(CURDIR)/libmumax2.so $(CURDIR)/../../lib
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu


.PHONY: clean
clean:
	rm -f *.o *.so

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

