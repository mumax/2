include $(GOROOT)/src/Make.inc

GOFILES=\
	main.go\

main: $(GOFILES)
	$(GC) -I_obj $(GOFILES)
	$(LD) -L_obj -o $@ *.$O

.PHONY:	clean
clean:
	rm -f  main *.a *.$O $(CLEANFILES)
