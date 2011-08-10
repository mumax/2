mumax2=../bin/mumax2

TESTFILES=\
		  test.py.out\
		  #Test.java.out\
		  
.PHONY: all
all:

.PHONY: test
test: $(TESTFILES)

.PHONY: $(TESTFILES)
$(TESTFILES): 
	$(mumax2) -f $(@:.out='')

.PHONY: install
install:

.PHONY: bench
bench:

.PHONY: clean
clean:
	rm -rf *.out *.log


