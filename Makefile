include src/Make.inc

all:
	$(MAKE) --no-print-directory --directory=src/libmumax2 
	go install mumax2-bin

.PHONY: clean
clean:
	make clean -C src/libmumax2
	go clean

.PHONY: test
test:
	echo todo	

.PHONY: doc
doc:
	make -C examples
	make -C doc
	ln -sf doc/manual/manual.pdf mumax2-manual.pdf

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 


