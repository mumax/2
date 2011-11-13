UNAME := $(shell uname)

# exclude muview for the moment in Os X
ifeq ($(UNAME), Darwin) # Os X
	dirs=\
		libmumax2\
		libomf\
		libtensor\
		pkg\
		mumax2\
		python\

else # Linux and other Unix based systems
	dirs=\
		libmumax2\
		libomf\
		libtensor\
		pkg\
		mumax2\
		muview\
		python\

endif

all: $(dirs)

pkg: libmumax2
mumax2: pkg
muview: libomf libtensor
python: mumax2 

include Dirs.pkg
