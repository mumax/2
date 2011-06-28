dirs=\
	common\
	host\
	gpu\
	engine\
	main\
	apigen\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
main: common engine apigen
apigen: common engine

include ../../Dirs.pkg


