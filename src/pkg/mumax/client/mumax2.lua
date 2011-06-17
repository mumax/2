
infifo=0
outfifo=0
initialzed=false

function init()
	-- create handshake file
	h = assert(io.open("handshake", "w"))
	h:close()


end

function call(cmd, args)
	if (not initialized) then
		init()
	end
	outfifo:write(cmd, args)
end

init()
