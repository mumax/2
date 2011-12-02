UNAME := $(shell uname)

# exclude muview for the moment in Os X
ifeq ($(UNAME), Darwin) # Os X
        dirs=\
                libmumax2\
                libomf\
                libtensor\
                pkg\
                mumax2\
                python\

else # Linux and other Unix based systems
        dirs=\
                libmumax2\
                libtensor\
                pkg\
                mumax2\
                python\
                #libomf\
                #muview\
        
endif

MODULES=$(wildcard *.mod)

all: $(dirs) $(MODULES)

pkg: libmumax2
mumax2: pkg $(MODULES)
muview: libomf libtensor
python: mumax2 

$(MODULES):
	$(MAKE) install --no-print-directory --directory=$@/pkg $(TARGET)
	
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
	$(MAKE) TARGET=clean
	rm -rf $(CLEANFILES)

