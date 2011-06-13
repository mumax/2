infifo=open('test.out/out.fifo', 'r')
outfifo=open('test.out/in.fifo', 'w')

print "Version"
outfifo.write('Version\n')
outfifo.flush()
print infifo.readline()

print "Echo 123"
outfifo.write('Echo 123\n')
outfifo.flush()
print infifo.readline()

print "Sum 1 2"
outfifo.write('Sum 1 2\n')
outfifo.flush()
print infifo.readline()
