dirs=\
	common\
	host\
	gpu\
	engine\
	main\
	apigen\
	queue\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
main: $(dirs)
apigen: common engine

include ../../Dirs.pkg


