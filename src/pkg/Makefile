.PHONY: main
main:
	make clean
	make pkgs
	make -C main

pkgs:
	make -C common
	make -C engine
	make -C refsh

test:	install
	make test -C common
	make test -C engine
	make test -C refsh

bench:	install
	make bench -C common
	make bench -C engine
	make bench -C refsh

install:
	make install -C common
	make install -C engine
	make install -C refsh

.PHONY: clean
clean:
	make clean -C common
	make clean -C engine
	make clean -C refsh
	make clean -C main
