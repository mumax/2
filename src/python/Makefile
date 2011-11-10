all: mumax2.py

mumax2.py: mumax2.py.tail 
	./apigen.sh $(CURDIR)/../pkg/mumax/engine/api.go
	mv mumax2.py mumax2.py.head
	cat mumax2.py.head mumax2.py.tail > mumax2.py
	rm -f mumax2.py.head

CLEANFILES=\
	mumax2.py\
	mumax2.pyc\
	mumax2.lua\
	Mumax2.java\
	Mumax2.class\


.PHONY: clean
clean:
	rm -rf $(CLEANFILES)

.PHONY: install
install: all

.PHONY: test
test: 

.PHONY: bench
bench:

