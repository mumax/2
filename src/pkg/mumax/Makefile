dirs=\
	common\
	host\
	gpu\
	engine\
	frontend\
	apigen\
	queue\
	main\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
frontend: common engine
apigen: common engine frontend
queue: common
main: $(dirs)

include ../../Dirs.pkg


