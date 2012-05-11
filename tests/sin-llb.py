from mumax2 import *
from mumax2_geom import *
from mumax2_magstate import *
from math import * 

Nx = 2048
Ny = 1
Nz = 1

setgridsize(Nx, Ny, Nz)
length=2048e-9
width=2e-9
thickness=2e-9
setcellsize(length/Nx, width/Ny, thickness/Nz)

# LLG 
load('micromagnetism')
load('solver/rk12')

# Py
setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('gamma', 2.211e5)

setv('dt', 0.001e-12)
setv('m_maxerror', 1./1000)

# The applied field is strong enough to saturate 
m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

# static applied field
Bx = 1.0 
By = 0.0 
Bz = 0.0 

b=makearray(3, Nx, Ny, Nz)

for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            b[0][ii][jj][kk] = 1
            b[1][ii][jj][kk] = 0
            b[2][ii][jj][kk] = 0
            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(3):
            b[0][ii][jj][kk] = 1
            b[1][ii][jj][kk] = 1
            b[2][ii][jj][kk] = 0
            

setmask('B_ext',b)

# Get the ground state

savegraph("graph.png")

run_until_smaller('maxtorque', 1e-5 * gets('gamma') * 800e3)

load('t-baryakhtar')

save("m","png",[])
save("m","ovf",[])

setv('alpha', 0.01)
setv('beta', 0.02)
setv('t', 0)
setv('dt', 1e-15)

save("B_ext","png",[])

tt = 5e-12 # 1/2*tt = bandwidth ~166 GHz

# Set up an rf applied field

N = 1024 # 2048 timesteps, ~6 ns
fcut = 30e9 # harmonic field frequency, Hz
Brf0 = 0.001 # 10 Oe = 0.001 T


for i in range(N):
        t = tt * i
        arg = 2 * pi * fcut * t
        Brfy = Brf0 * sin(arg)
        setpointwise('B_ext', t, [Bx, Brfy, 0])
	
TT = N * tt

autosave("m", "png", [], tt)
autosave("m", "ovf", [], tt)
autotabulate(["t", "<m>"], "m.txt", tt)
autotabulate(["t", "<B_ext>"], "B_ext.txt", tt)

run(5e-9)

quit()
