mumax2=../bin/mumax2

TESTFILES=\
		  test.py.out\
		  Test.java.out\
		  
.PHONY: all
all:

.PHONY: test
test: $(TESTFILES)

.PHONY: $(TESTFILES)
$(TESTFILES): 
	$(mumax2) -force $(@:.out='')
	$(mumax2) -listen -timeout 2s & sleep 1s; $(mumax2) -server localhost -force $(@:.out='')

.PHONY: install
install:

.PHONY: bench
bench:

.PHONY: clean
clean:
	rm -rf *.out *.log


