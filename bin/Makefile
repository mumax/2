all: apigen


apigen:
	./mumax2 -apigen
	python mumax2.py
	javac Mumax2.java
	lua mumax2.lua
	#g++ mumax2.h

CLEANFILES=\
	mumax2-client\
	mumax2.py\
	mumax2.pyc\
	Mumax2.java\
	Mumax2.class\


.PHONY: clean
clean:
	rm -rf $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test:
