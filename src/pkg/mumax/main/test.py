infifo=open('test.out/out.fifo', 'r')
outfifo=open('test.out/in.fifo', 'w')

outfifo.write('line1\n')
outfifo.write('line2\n')
outfifo.write('line3\n')
outfifo.write('line4\n')
