dirs=\
	src\
	bin\
	tests\

CLEANFILES+=*.log

all: $(dirs) githooks

bin: src

tests: bin

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 

include src/Dirs.pkg

