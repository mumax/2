.PHONY: main
main:
	make clean
	make pkgs
	make -C main

pkgs:
	make -C common
	make -C refsh
	make -C engine

test:	install
	make test -C common
	make test -C refsh
	make test -C engine

bench:	install
	make bench -C common
	make bench -C refsh
	make bench -C engine

install:
	make install -C common
	make install -C refsh
	make install -C engine

.PHONY: clean
clean:
	make clean -C common
	make clean -C refsh
	make clean -C engine
	make clean -C main
