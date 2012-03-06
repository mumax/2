export GOPATH=$(CURDIR)/../..

DIRS=apigen common engine frontend gpu host modules ovf queue

all: $(DIRS)

.PHONY: $(DIRS)
$(DIRS): 
		go install -v mumax/$@
