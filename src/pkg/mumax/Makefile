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

apigen: client

main: client apigen

