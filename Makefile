include src/Make.inc

export GOPATH=$(CURDIR)

all:
	$(MAKE) --no-print-directory --directory=src/libmumax2 
	go run src/cuda/setup-cuda-paths.go -dir=src/cuda/
	go install -v --gccgoflags '-Ofast -march=native' mumax2-bin
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


