from mumax2 import *
from mumax2_geom import *
from mumax2_magstate import *
from math import * 
Nx = 1024
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

setv('dt', 0.001e-12)
setv('m_maxerror', 1./1000)

# The applied field is strong enough to saturate 
m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

# static applied field
Bx = 1.0 / mu0 # 10kOe = 1 T
By = 0.0 / mu0 # T
Bz = 0.0 / mu0 # T  
setv('B_ext', [Bx, By, Bz])

# Get the ground state

savegraph("graph.png")

run_until_smaller('maxtorque', 1e-6 * gets('gamma') * 800e3)

save("m","png",[])
save("m","ovf",[])

setv('alpha', 0.01)

setv('t', 0)
setv('dt', 0.001e-12)


add_to('H_eff','Brf')

rfmask = makearray(3, Nx, Ny, Nz)
rfmask[0][0][0][0] = 0
rfmask[1][0][0][0] = 1
rfmask[2][0][0][0] = 0
rfmask[0][1][0][0] = 0
rfmask[1][1][0][0] = 1
rfmask[2][1][0][0] = 0
rfmask[0][2][0][0] = 0
rfmask[1][2][0][0] = 1
rfmask[2][2][0][0] = 0
setmask('Brf',rfmask)

save("Brf","png",[])

tt = 3e-12 # 1/2*tt = bandwidth ~166 GHz

# Set up an rf applied field

N = 2048 # 2048 timesteps, ~6 ns
fcut = 100e9 # sinc cutoff, Hz
Brf0 = 0.1 # 1000 Oe = 0.1 T


for i in range(N):
        t = tt * i
        arg = 2 * pi * fcut * t
        if arg != 0.0:
            By = Brf0 * sin(arg) / arg
        else:
            By = Brf0
        setpointwise('Brf', t, [0, By / mu0, 0])
	
TT = N * tt

autosave("m", "png", [], tt)
autosave("m", "ovf", [], tt)
autotabulate(["t", "<m>"], "m.txt", tt)

run(6e-9)

quit()
