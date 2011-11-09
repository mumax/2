include src/Make.inc

dirs=\
	src\
	tests\
	#doc\

CLEANFILES+=*.log

all: $(dirs) githooks

tests: src lib
test: src lib

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 

include src/Dirs.pkg


