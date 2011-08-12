include src/Make.inc

dirs=\
	src\
	tests\

CLEANFILES+=*.log

all: $(dirs) githooks


tests: src
test: src

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 

include src/Dirs.pkg

