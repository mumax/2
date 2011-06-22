dirs=\
	common\
	host\
	gpu\
	engine\
	main\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
main: common engine

include ../../Dirs.pkg


