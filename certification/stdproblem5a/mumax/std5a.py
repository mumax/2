from mumax2 import *
from mumax2_geom import *
from math import * 

Nx = 64
Ny = 32
Nz = 2

sX = 160e-9
sY = 80e-9
sZ = 5e-9
 
setgridsize(Nx, Ny, Nz)
setcellsize(sX/Nx, sY/Ny, sZ/Nz)


load('micromagnetism')
load('solver/rk12')

savegraph("graph.png")


setv('Msat', 800e3)
# setmask('Msat', ellipse())
setv('Aex', 13e-12)

setv('alpha', 1.0)

setv('dt', 1e-15)
setv('maxdt', 1e-12)
setv('m_maxerror', 1./5000)

# static applied field
Bx = 0.0 # 100 Oe 
By = 0.0 
Bz = 0.0 

B=[ [[[1]]], [[[0]]], [[[0]]] ]
setmask("B_ext", B)
setv('B_ext', [Bx, By, Bz])

save("B_ext","ovf",[])

# Set a initial magnetisation to C-state
mv = makearray(3, Nx, Ny, Nz)

for m in range(Nx):
    for n in range(Ny):
        for o in range(Nz):
		
            xx = float(m)/float(Nx)
            mv[0][m][n][o] = 1.0
            mv[1][m][n][o] = 0.0
            mv[2][m][n][o] = 0.0
            
            if (xx < 0.25) :	
                mv[0][m][n][o] =  0.0
                mv[1][m][n][o] = -1.0
                mv[2][m][n][o] = -0.1
            if (xx > 0.75):
                mv[0][m][n][o] =  0.0
                mv[1][m][n][o] = -1.0
                mv[2][m][n][o] = -0.1
                
setarray('m', mv)

save("m","png",[])
save("m","vtk",[])

run_until_smaller('maxtorque', 1e-5 * gets('gamma') * 800e3)

setv('alpha', 0.01)

setv('dt', 1e-15)
setv('t', 0)

load('slonczewski')

t_fl = makearray(1,Nx,Ny,Nz)
for m in range(Nx):
    for n in range(Ny):
        for o in range(Nz):
            t_fl[0][m][n][o] = 1.0
setmask('t_fl', t_fl)
setv('t_fl', sZ)

setv('lambda',1.0)
setv('Pol',0.5669)
setv('epsilon_prime', 0.0)

save("m","ovf",[])
save("m","png",[])
save("m","vtk",[])

setv("t", 0.0)
setv("dt", 1e-15)


pdeg = 1    
prad = pdeg * pi / 180.0
px = cos(prad)
py = sin(prad)

J = -0.008  # total current in amps

carea = sX * sY
jc = J / carea  

print "Current density is: " +  str(jc) + "\n"
 
j=[ [[[0]]], [[[0]]], [[[1]]] ]
setmask('j', j)
setv('j', [0, 0, jc])
save("j","ovf",[])

p=[ [[[1]]], [[[1]]], [[[0]]] ]
setmask('p', p)
setv('p', [px, py, 0])
save("p","ovf",[])

autosave("m", "png", [], 1e-12)
autotabulate(["t", "<m>"], "m.txt", 1e-12)

run(5e-9)
printstats()
sync()
