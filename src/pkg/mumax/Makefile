dirs=\
	common\
	host\
	gpu\
	engine\
	frontend\
	apigen\
	queue\

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
frontend: common engine gpu
apigen: common engine frontend
queue: common

include ../GoDirs.pkg

