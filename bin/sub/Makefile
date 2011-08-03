all: apigen

apigen:
	./mumax2-apigen

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
install:

.PHONY: test
test: apigen
	python mumax2.py
	#javac Mumax2.java
	#lua mumax2.lua
	#g++ mumax2.h
