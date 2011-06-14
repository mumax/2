dirs=\
	src\
	bin\

all: $(dirs) githooks

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/git/pre-commit .git/hooks 

include src/Dirs.pkg

