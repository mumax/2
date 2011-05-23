pkgs=\
	cuda\
	mumax\

mumax: cuda

.PHONY: all $(pkgs)
all: $(pkgs)

$(pkgs):
	$(MAKE) --directory=$@ $(TARGET)


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

