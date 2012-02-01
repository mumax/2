mumax2=../bin/mumax2

TESTFILES=\
	pointwise.py.out\
	coulomb.py.out\
	dipole.py.out\
	addto.py.out\
	current.py.out\
		  
.PHONY: all
all:

.PHONY: test
test: $(TESTFILES)

.PHONY: $(TESTFILES)
$(TESTFILES): 
	$(mumax2) -f -gpu=all $(@:.out='')
	$(mumax2) -f -gpu=0,0 $(@:.out='')
	$(mumax2) -f -gpu=0 $(@:.out='')

.PHONY: install
install:

.PHONY: bench
bench:

.PHONY: clean
clean:
	rm -rf *.out *.log


