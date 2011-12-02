include ../Make.inc

CUFILES= $(wildcard *.cu ../*.module/*.cu)

HFILES=$(CUFILES:.cu=.h)

OFILES=$(CUFILES:.cu=.o)

all: libmumax2.so libmumax2.h

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu) -o $(@)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu

libmumax2.h: $(HFILES)
	cp libmumax2.h.head libmumax2.h
	$(foreach hfile, $(HFILES),  echo \#include "\""$(hfile)"\"" >> libmumax2.h;)
	cat libmumax2.h.tail >> libmumax2.h	

.PHONY: clean
clean:
	rm -f *.o *.so libmumax2.h

install: all

.PHONY: test
test:

.PHONY: bench
bench:

