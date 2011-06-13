infifo=open('test.out/out.fifo', 'r')
outfifo=open('test.out/in.fifo', 'w')

print "send"
outfifo.write('Version\n')
outfifo.flush()
print "got", infifo.readline()

print "send"
outfifo.write('Echo 123\n')
outfifo.flush()
print "got", infifo.readline()

print "send"
outfifo.write('Sum 1 2\n')
outfifo.flush()
print "got", infifo.readline()
