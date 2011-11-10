include ../Make.inc

OFILES=\
	tensor.o\

all: libtensor.a

$(OFILES): %.o: %.cpp %.h
	 g++ -c $(@:.o=.cpp)

libtensor.a: $(OFILES) 
	ar -cvq libtensor.a $(OFILES) 

.PHONY: clean
clean:
	rm -f *.o *.a

install: all

.PHONY: test
test:

.PHONY: bench
bench:

