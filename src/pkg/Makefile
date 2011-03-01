all:
	make -C common
	make -C engine

test:	install
	make test -C common
	make test -C engine

bench:	install
	make bench -C common
	make bench -C engine

install:
	make install -C common
	make install -C engine

.PHONY: clean
clean:
	make clean -C common
	make clean -C engine
