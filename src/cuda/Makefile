export GOPATH=$(CURDIR)/../..

DIRS=driver runtime cufft curand 

all: $(DIRS)

.PHONY: $(DIRS)
$(DIRS): 
		go install -v cuda/$@

