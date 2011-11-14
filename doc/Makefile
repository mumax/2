.SILENT:

godirs=\
	mumax\
	mumax/common\
	mumax/gpu\
	cuda\
	cuda/driver\

all: doxygen
# $(godirs)

doxygen:
	doxygen Doxyfile||echo "make doc requires package doxygen, graphviz"

.PHONY: $(godirs)
$(godirs):
	mkdir -p html/$@
	#ln -sf $(shell readlink -f ../src/pkg/)/$@ $(GOROOT)/src/pkg/$@
	godoc -path=. -html $@ > html/$@/index.html

cu:

.PHONY: clean
clean:
	rm -rf html latex

.PHONY: install
test:

.PHONY: test
test:

.PHONY: bench
test:

