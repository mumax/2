include ../../Make.inc

CUFILES   = $(wildcard *.cu)  
HFILES    = $(CUFILES:.cu=.h) 
OFILES    = $(CUFILES:.cu=.o) 


all: $(OFILES)

$(OFILES): %.o: %.cu %.h
	$(NVCC) -c -I../../libmumax2 $(@:.o=.cu) -o $(@)

.PHONY: clean
clean:
	rm -f *.o 

