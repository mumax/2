include ../Make.inc

CUFILES= $(wildcard *.cu ../*.module/*.cu)

OFILES=$(CUFILES:.cu=.o)

all: libmumax2.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu) -o $(@)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu

.PHONY: clean
clean:
	rm -f *.o *.so

install: all

.PHONY: test
test:

.PHONY: bench
bench:

