include src/Make.inc
export GOPATH=$(CURDIR)

all:
	$(MAKE) --no-print-directory --directory=src/libmumax2 
	go install mumax2-bin
	go install apigen
	go install texgen

.PHONY: clean
clean:
	make clean -C src/libmumax2
	rm -rf pkg/*
	rm -rf bin/mumax2-bin
	rm -rf bin/apigen
	rm -rf bin/texgen

.PHONY: test
test:
	echo todo	

.PHONY: doc
doc:
	#make -C examples
	make -C doc
	ln -sf doc/manual/manual.pdf mumax2-manual.pdf

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 


