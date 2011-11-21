from mumax2 import *
from sys import exit 

def fail(msg):
		echo("FAIL:" + str(msg))
		exit(42)

Nx = 128
Ny = 64
Nz = 2
setgridsize(Nx, Ny, Nz)

Sx = 5e-9
Sy = 5e-9
Sz = 10e-9
setcellsize(Sx, Sy, Sz)

load('micromagnetism')
savegraph("graph.dot")

setscalar('alpha', 0.1)
if getscalar('alpha') != 0.1:
		fail(getscalar('alpha'))

setscalar('Msat', 800e3)
if getscalar('Msat') != 800e3:
		fail(getscalar('Msat'))

setscalar('Aex', 12e-13)

m=[ [[[0]]], [[[0]]], [[[2]]] ]
setarray('m', m)

X=3
Y=2
Z=1
mset = [2, 0, 0]
mnorm = [1, 0, 0]
setcell('m', X,Y,Z, mset)
m=getcell('m', X,Y,Z)

for i in m:
	if m[i] != mnorm[i]:
		fail(str(i) + " " + str(m[i]))

m=getcell('m', 0,0,0)
if m != [0.,0.,1.]:
		fail(str(m))

printstats()


