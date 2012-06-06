from mumax2 import *

# Standard Problem 4

Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('demag_acc', 7)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-12)
setv('m_maxerror', 1./1000)
new_maxabs("my_maxtorque", "torque")
new_maxnorm("maxnorm", "torque")

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

t1=getv("maxtorque")
t2=getv("my_maxtorque")
t3=getv("maxnorm")
echo("maxtorque:" + str(t1) + " my_maxtorque:" + str(t2) + " maxnorm:" + str(t3))
if t3 != t1:
	crash
if t3 < t2:
	crash

new_maxabs("maxtorquez", "torque.z")
getv("maxtorquez")

printstats()
savegraph("graph.png")
