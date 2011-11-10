dirs=\
	libmumax2\
	libomf\
	libtensor\
	pkg\
	mumax2\
	muview\
	python\
	ptx\
	
all: $(dirs)

pkg: libmumax2
mumax2: pkg
muview: libomf libtensor
python: mumax2 

include Dirs.pkg
