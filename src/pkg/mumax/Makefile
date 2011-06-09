dirs=\
	common\
	host\
	gpu\
	client\
	main\

all: $(dirs)

include ../../Dirs.pkg

host: common

gpu: common host

client: common gpu host

main: client

