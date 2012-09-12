import numpy as np
from pylab import *

f = open('v.txt', 'w')
f.write("#m"+ "\t" + "v"+"\n")

for mag in range (40):
  sizeX=4800e-9
  data=np.loadtxt('./simulations/nanowire_H'+str(mag)+'.py.out/'+'m_H'+str(mag/10000.)+'_.txt')
  x_pos=sizeX/2*(1+data[:,1])
  v=(x_pos[1:]-x_pos[:-1])/(data[1:,0]-data[:-1,0])
  vg = float(sum(v[-30:-1])) / len(v[-30:-1])
  f.write(str(mag)+'\t'+str(vg) +"\n")
f.close()

data=np.loadtxt('v.txt')
plot(data[:,0]/10.,data[:,1])
xlabel('B [mT]')
ylabel('v [m/s]')


savefig('v.png')
close()
show()

