include ../Make.inc

CUFILES   = $(filter-out $(wildcard $(MY_CUFILES_WERRORLESS)), $(wildcard *.cu))  # my cuda files
CPPFILES     = $(wildcard *.cpp)                 # my c++ files
CU_HFILES    = $(CUFILES:.cu=.h)                 # cuda only header files
CU_HFILES   += $(MY_CUFILES_WERRORLESS:.cu=.h)                 # cuda only header files
HFILES       = $(CUHFILES) $(CPPFILES:.cpp=.h)   # all header files

ifndef SystemRoot
CU_OFILES    = $(CUFILES:.cu=.o)              	# my cuda objects
CPP_OFILES   = $(CPPFILES:.cpp=.o)              # c++ objects
else
CU_OFILES    = $(CUFILES:.cu=.obj)              
CPP_OFILES   = $(CPPFILES:.cpp=.obj)  
endif

OFILES       = $(CU_OFILES) $(CPP_OFILES)\
			   $(CU_OFILES_WERRORLESS)           # all objects


ifndef SystemRoot

all: libmumax2.so libmumax2.h

$(CU_OFILES): %.o: %.cu %.h
	$(NVCC) -dc -I. $(@:.o=.cu) -o $(@)

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
	
else

all: libmumax2.lib libmumax2.h

$(CU_OFILES): %.obj: %.cu %.h
	$(NVCC) --compiler-options -D_USRDLL --compiler-options -DDLL_EXPORTS -dc -I. $(@:.obj=.cu) -o $(@)

$(CPP_OFILES): %.obj: %.cpp %.h
	 $(CC) /D_USRDLL /DDLL_EXPORTS -c $(@:.obj=.cpp) -o $(@)

libmumax2.lib: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) --shared --linker-options -DLL $(OFILES) -o libmumax2.dll
	cp libmumax2.dll ../../bin/
	
libmumax2.h: $(CU_HFILES) multigpu.h
	cp libmumax2.h.head libmumax2.h.bak
	$(foreach hfile, $(CU_HFILES), echo #include "$(hfile)" >> libmumax2.h.bak)
	echo #include "multigpu.h" >> libmumax2.h.bak
	cat libmumax2.h.tail >> libmumax2.h.bak
	cat libmumax2.h.bak | sed "s/echo /\n/g" >> libmumax2.h
	
endif
	
.PHONY: link
link:
	ln -sf $(CURDIR)/libmumax2.so ../pkg/mumax/gpu
	$(foreach module, $(wildcard ../*.mod), ln -sf $(CURDIR)/libmumax2.so $(module)/pkg;)

.PHONY: clean
clean:
ifndef SystemRoot
	rm -f *.o *.so libmumax2.h ../*.mod/cuda/*.o
else
	rm -f *.def *.exp *.obj *.lib *.dll *.pdb libmumax2.h 
	rm -f -R ../*.mod/cuda/*.obj
endif

install: all

.PHONY: test
test:

.PHONY: bench
bench:

