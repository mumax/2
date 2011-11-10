include ../Make.inc

OFILES=\
	OMFHeader.o\
	OMFImport.o\

all: libomf.a

$(OFILES): %.o: %.cpp %.h
	 $(CC) -c $(@:.o=.cpp)

libomf.a: $(OFILES) 
	ar -cvq libomf.a $(OFILES) 

.PHONY: clean
clean:
	rm -f *.o *.a

install: all

.PHONY: test
test:

.PHONY: bench
bench:

