dirs=\
	src\
	bin\
	tests\

all: $(dirs) githooks

bin: src

tests: bin

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/git/pre-commit .git/hooks 

include src/Dirs.pkg

