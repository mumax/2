include src/Make.inc

dirs=\
	src\
	lib\
	tests\

CLEANFILES+=*.log

all: $(dirs) githooks


tests: src lib
test: src lib
lib: src

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 

include src/Dirs.pkg

