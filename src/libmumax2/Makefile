include ../Make.inc

MY_CUFILES= $(wildcard *.cu)                    # my cuda files
MOD_CUFILES= $(wildcard ../*.modules/cuda/*.cu) # module cuda files
CUFILES=$(MY_CUFILES) $(MOD_CUFILES)            # all cuda files
CPPFILES= $(wildcard *.cpp)                     # my c++ files

CU_HFILES=$(CUFILES:.cu=.h)                     # cuda only header files
HFILES=$(CUHFILES) $(CPPFILES:.cpp=.h)          # all header files

MY_CU_OFILES=$(MY_CUFILES:.cu=.o)               # my cuda objects
MOD_CU_OFILES=$(MOD_CUFILES:.cu=.o)             # module cuda objects
CU_OFILES=$(MY_CU_OFILES) $(MOD_CU_OFILES)      # all cuda objects
CPP_OFILES=$(CPPFILES:.cpp=.o)                  # c++ objects
OFILES=$(CU_OFILES) $(CPP_OFILES)               # all objects


all: libmumax2.so libmumax2.h

$(CU_OFILES): %.o: %.cu %.h
	$(NVCC) -c -I. $(@:.o=.cu) -o $(@)


$(CPP_OFILES): %.o: %.cpp %.h
	 $(CC) -fPIC -c $(@:.o=.cpp) -o $(@)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu

libmumax2.h: $(CU_HFILES) multigpu.h
	cp libmumax2.h.head libmumax2.h
	$(foreach hfile, $(CU_HFILES),  echo \#include "\""$(hfile)"\"" >> libmumax2.h;)
	echo \#include \"multigpu.h\" >> libmumax2.h
	cat libmumax2.h.tail >> libmumax2.h	

.PHONY: clean
clean:
	rm -f *.o *.so libmumax2.h ../*.modules/cuda/*.o

install: all

.PHONY: test
test:

.PHONY: bench
bench:

