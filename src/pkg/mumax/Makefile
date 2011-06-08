dirs=\
	common\
	host\
	gpu\
	client\
	apigen\
	main\

include ../../Dirs.pkg

host: common

gpu: common host

client: common gpu host

apigen: common client

main: client apigen

