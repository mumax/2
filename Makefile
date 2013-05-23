include src/Make.inc

export GOPATH=$(CURDIR)

ifndef SystemRoot
LIBNAME=libmumax2.so
export CUDAROOT=/usr/local/cuda-5.0
export NVROOT=/usr/lib64/nvidia
export CUDA_INC_PATH=$(CUDAROOT)/include/
export CUDA_LIB_PATH=$(NVROOT)/opengl/lib64/:$(CUDAROOT)/lib64/
else
LIBNAME=libmumax2.dll
endif

all:
	$(MAKE) --no-print-directory --directory=src/libmumax2 
	cp src/libmumax2/$(LIBNAME) src/mumax/gpu/
	cp src/libmumax2/$(LIBNAME) .
	go run src/cuda/setup-cuda-paths.go -dir=src/cuda/
	go install -v mumax2-bin
	go install -v apigen
	go install -v texgen
	go install -v template
	make -C src/python
ifndef SystemRoot	
	make -C src/libomf
	make -C src/muview
endif

.PHONY: clean
clean:	
	rm -rf pkg/*
	rm -rf src/mumax/gpu/$(LIBNAME)
	rm $(LIBNAME)
ifndef SystemRoot	
	rm -rf bin/mumax2-bin
	rm -rf bin/apigen
	rm -rf bin/texgen
else
	rm -rf bin/mumax2-bin.exe
	rm -rf bin/apigen.exe
	rm -rf bin/texgen.exe
	rm -rf bin/libmumax2.dll
endif	
	make clean -C src/python
	make clean -C src/libmumax2
	make clean -C src/libomf
	make clean -C src/muview
.PHONY: test
test:
	echo todo
		
.PHONY: tidy	
tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

.PHONY: love	
love:
	@echo Oh, yeah
	@echo Do it again to MuMax2!
	
.PHONY: doc
doc:

	make -C doc
	ln -sf doc/manual/manual.pdf mumax2-manual.pdf

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 


