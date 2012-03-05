dirs=\
	common\
	host\
	gpu\
	engine\
	modules\
	frontend\
	apigen\
	ovf\
	queue\

all: $(dirs)

host: common
gpu: common host
engine: common gpu host
frontend: common engine gpu
apigen: common engine frontend
ovf: common host engine
queue: common
modules: common engine gpu

include ../GoDirs.pkg

