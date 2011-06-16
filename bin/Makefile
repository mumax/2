all: apigen


apigen:
	./mumax2 -apigen
	python mumax2.py
	javac Mumax2.java
	#g++ mumax2.h

.PHONY: clean
clean:
	rm -rf mumax2-client

.PHONY: install
install:

.PHONY: test
test:
