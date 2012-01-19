mumax2=../bin/mumax2

TESTFILES=\
		  access.py.out\
		  demag.py.out\
		  engine.py.out\
		  micromag.py.out\
		  pointwise.py.out\
		  coulomb.py.out\
		  #anisotropy.py.out\
		  #Test.java.out\
		  #api.py.out\
		  
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


