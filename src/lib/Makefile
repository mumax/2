OFILES=\
	reduce.o\

NVCC=nvcc 
NVCC+=\
  --compiler-options -Wall\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
#--use_fast_math\

all: $(ofiles)

$(ofiles): %.o: %.cu %.h
	 $(NVCC) -c $(@:.o=.cu)

libmumax2.so: $(ofiles) libmumax2.h
	$(NVCC) -shared $(OFILES) $(CUDALIBS) $(L) -o libgpukern.so
	cp -f $(CURDIR)/libgpukern.so $(CURDIR)/../../lib
.PHONY: clean
clean:
	rm -f *.o

.PHONY: install
install:

.PHONY: test
test:

.PHONY: bench
bench:

