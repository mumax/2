all: mumax2.py

mumax2.py: mumax2.py.tail mumax2-apigen
	./mumax2-apigen
	mv mumax2.py mumax2.py.head
	cat mumax2.py.head mumax2.py.tail > mumax2.py

CLEANFILES=\
	mumax2-apigen\
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
test: mumax2.py
	python mumax2.py
	#javac Mumax2.java
	#lua mumax2.lua
	#g++ mumax2.h

.PHONY: bench
bench:

