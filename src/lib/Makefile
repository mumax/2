OFILES=\
	gpu_safe.o\
	gpu_conf.o\
	gpu_properties.o\
	add.o\
	reduce.o\


#HFILES=$(OFILES:.o=.h)

NVCC=nvcc 
NVCC+=\
  --compiler-options -Wall\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
#--use_fast_math\

all: libmumax2.so

$(OFILES): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmumax2.so: $(OFILES) libmumax2.h
	$(NVCC) -shared $(OFILES) $(CUDALIBS) $(L) -o libgpukern.so
	cp -f $(CURDIR)/libgpukern.so $(CURDIR)/../../lib

#libmumax2.h: $(HFILES)
#	rm -f libmumax2.h
#	ls *.h > libmumax2.h

.PHONY: clean
clean:
	rm -f *.o

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

