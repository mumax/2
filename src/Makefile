dirs=\
	libmumax2\
	libtensor\
	pkg\
	mumax2\
	muqueue\
	python\

# exclude muview for the moment in Os X
UNAME := $(shell uname)
ifneq ($(UNAME), Darwin) # Linux and other Unix based systems

dirs+=\
	  #libomf\
      #muview\
        
endif


MODULES=$(wildcard *.mod)

all: $(dirs) $(MODULES)

$(MODULES): libmumax2 pkg
pkg: libmumax2
mumax2: pkg $(MODULES)
muview: libomf libtensor
python: mumax2 
muqueue: mumax2

$(MODULES):
	$(MAKE) install --no-print-directory --directory=$@/pkg
	
.PHONY: $(dirs)

$(dirs):
	$(MAKE) --no-print-directory --directory=$@ $(TARGET)


.PHONY: install
install:
	$(MAKE) TARGET=install


.PHONY: test
test: install
	$(MAKE) TARGET=test


.PHONY: bench
bench: install
	$(MAKE) TARGET=bench


.PHONY: clean
clean:
	#$(MAKE) TARGET=clean
	rm -rf $(CLEANFILES)
	$(foreach module, $(MODULES), make clean -C $(module)/pkg;)
	$(foreach dir, $(dirs), make clean -C $(dir);)

