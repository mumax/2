infifo=open('test.out/out.fifo', 'r')
outfifo=open('test.out/in.fifo', 'w')

outfifo.write('line1 arg1\n')
outfifo.write('line2 arg1 arg2\n')
outfifo.write('line3 arg1 arg2 arg3\n')
