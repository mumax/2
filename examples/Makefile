mumax2=../bin/mumax2

TESTFILES=$(wildcard *.py)
		  
.PHONY: all
all: $(TESTFILES)

.PHONY: $(TESTFILES)
$(TESTFILES): 
	$(mumax2) -f  $(@)


.PHONY: clean
clean:
	rm -rf *.out *.log


