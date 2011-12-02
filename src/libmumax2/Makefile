include ../Make.inc

CUFILES= $(wildcard *.cu ../*.module/*.cu)

CPPFILES= $(wildcard *.cpp)

CUHFILES=$(CUFILES:.cu=.h)

HFILES=$(CUHFILES) $(CPPFILES:.cpp=.h)

CUOFILES=$(CUFILES:.cu=.o) 
CPPOFILES=$(CPPFILES:.cpp=.o)
OFILES=$(CUOFILES) $(CPPOFILES)


all: libmumax2.so libmumax2.h

$(CUOFILES): %.o: %.cu %.h
	$(NVCC) -c $(@:.o=.cu) -o $(@)

$(CPPOFILES): %.o: %.cpp %.h
	 $(CC) -fPIC -c $(@:.o=.cpp) -o $(@)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu

libmumax2.h: $(CUHFILES)
	cp libmumax2.h.head libmumax2.h
	$(foreach hfile, $(CUHFILES),  echo \#include "\""$(hfile)"\"" >> libmumax2.h;)
	cat libmumax2.h.tail >> libmumax2.h	

.PHONY: clean
clean:
	rm -f *.o *.so libmumax2.h

install: all

.PHONY: test
test:

.PHONY: bench
bench:

