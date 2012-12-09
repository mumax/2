include ../Make.inc

OFILES=\
	libdump.o\

all: libdump.a

$(OFILES): %.o: %.c %.h
	 $(CC) -c $(@:.o=.c)

libdump.a: $(OFILES) 
	ar -cvq libdump.a $(OFILES) 

.PHONY: clean
clean:
	rm -f *.o *.a

install: all

.PHONY: test
test:

.PHONY: bench
bench:

