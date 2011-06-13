infifo=open('test.out/out.fifo', 'r')
outfifo=open('test.out/in.fifo', 'w')

def call(command, args):
	outfifo.write(command)
	for a in args:
			outfifo.write(" ")
			outfifo.write(str(a))
	outfifo.write("\n")
	outfifo.flush()
	return infifo.readline()


print "Version", call("Version", [])
print "Echo 123", call("Echo", [123])
print "Sum 1 2", call("Sum", [1,2])

#print "Echo 123"
#outfifo.write('Echo 123\n')
#outfifo.flush()
#print infifo.readline()
#
#print "Sum 1 2"
#outfifo.write('Sum 1 2\n')
#outfifo.flush()
#print infifo.readline()
