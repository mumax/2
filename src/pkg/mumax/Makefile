dirs=\
	common\
	host\
	gpu\
	engine\
	client\
	main\

CLEANFILES+=*.log

all: $(dirs)

host: common
gpu: common host
client: common gpu host engine
main: client engine

include ../../Dirs.pkg


