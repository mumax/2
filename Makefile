include src/Make.inc

dirs=\
	src\
	tests\

CLEANFILES+=*.log

all: $(dirs) githooks

.PHONY: doc
doc:
	make -C examples
	make -C doc
	ln -sf doc/manual/manual.pdf mumax2-manual.pdf

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 

include src/Dirs.pkg

