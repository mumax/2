include ../Make.inc

MY_CUFILES_WERRORLESS = random.cu				 		# my cuda files to be compiled without Werror flag
CU_OFILES_WERRORLESS  = $(MY_CUFILES_WERRORLESS:.cu=.o) # my cuda object compiled without Werror flag

MY_CUFILES   = $(filter-out $(wildcard $(MY_CUFILES_WERRORLESS)), $(wildcard *.cu))  # my cuda files
#MY_CUFILES   = $(wildcard *.cu)					 # my cuda files
MOD_CUFILES  = $(wildcard ../*.mod/cuda/*.cu)    # module cuda files
CUFILES      = $(MY_CUFILES) $(MOD_CUFILES)      # all cuda files
CPPFILES     = $(wildcard *.cpp)                 # my c++ files
CU_HFILES    = $(CUFILES:.cu=.h)                 # cuda only header files
CU_HFILES   += $(MY_CUFILES_WERRORLESS:.cu=.h)                 # cuda only header files
HFILES       = $(CUHFILES) $(CPPFILES:.cpp=.h)   # all header files
MY_CU_OFILES = $(MY_CUFILES:.cu=.o)              # my cuda objects
MOD_CU_OFILES= $(MOD_CUFILES:.cu=.o)             # module cuda objects
CU_OFILES    = $(MY_CU_OFILES) $(MOD_CU_OFILES)  # all cuda objects
CPP_OFILES   = $(CPPFILES:.cpp=.o)               # c++ objects
OFILES       = $(CU_OFILES) $(CPP_OFILES)\
			   $(CU_OFILES_WERRORLESS)           # all objects


all: libmumax2.so libmumax2.h

$(CU_OFILES): %.o: %.cu %.h
	$(NVCC) -c -I. $(@:.o=.cu) -o $(@)

$(CU_OFILES_WERRORLESS): %.o: %.cu %.h
	$(NVCCWERRORLESS) -c -I. $(@:.o=.cu) -o $(@)

$(CPP_OFILES): %.o: %.cpp %.h
	 $(CC) -fPIC -c $(@:.o=.cpp) -o $(@)

libmumax2.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libmumax2.so

libmumax2.h: $(CU_HFILES) multigpu.h
	cp libmumax2.h.head libmumax2.h
	$(foreach hfile, $(CU_HFILES),  echo \#include "\""$(hfile)"\"" >> libmumax2.h;)
	echo \#include \"multigpu.h\" >> libmumax2.h
	cat libmumax2.h.tail >> libmumax2.h	

.PHONY: link
link:
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu
	$(foreach module, $(wildcard ../*.mod), ln -sf $(CURDIR)/libmumax2.so $(module)/pkg;)

.PHONY: clean
clean:
	rm -f *.o *.so libmumax2.h ../*.mod/cuda/*.o

install: all

.PHONY: test
test:

.PHONY: bench
bench:

