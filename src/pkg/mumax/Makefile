dirs=\
	common\
	host\
	gpu\
	engine\
	frontend\
	apigen\
	queue\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
frontend: common engine gpu
apigen: common engine frontend
queue: common

include ../../Dirs.pkg

