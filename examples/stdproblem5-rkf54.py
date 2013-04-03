## An example to demonstrate current induced switching in Py nanopraticle
# @author Mykola Dvornik

# import mumax Python modules
from mumax2 import *
from mumax2_geom import *
from math import * 


# Define mesh
Nx = 64
Ny = 32
Nz = 2

# Define sample sizes, m

sX = 160e-9
sY = 80e-9
sZ = 5e-9
 
setgridsize(Nx, Ny, Nz)
setcellsize(sX/Nx, sY/Ny, sZ/Nz)

# load basic modules for micromagnetic simulations

load('micromagnetism')
load('solver/rkf54')

savegraph("graph.png")

# Define saturation magnetization, A/m
setv('Msat', 800e3)
# Define exchange constant, J/m^2
setv('Aex', 13e-12)
# Define Gilbert damping constant
setv('alpha', 1.0)

# Define the initial time step for the solver, s
setv('dt', 1e-15)
# Define the maximum time step solver can take, s
setv('maxdt', 1e-12)
# Define maximum magnetization error between steps 
setv('m_maxerror', 1e-4)
# Define relative magnetization error between steps 
setv('m_relerror', 1e-3)

# Define static applied field, T
Bx = 0.0 # 100 Oe 
By = 0.0 
Bz = 0.0 
setv('B_ext', [Bx, By, Bz])

# Define spatial mask of the applied field, uniform in this case.
B=[ [[[1]]], [[[0]]], [[[0]]] ]
setmask("B_ext", B)
# Save the applied field to the OVF2.0 file for debugging
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

# Save initial magnetization to PNG and VTK files.
save("m","png",[])
save("m","vtk",[])

# Run the solver, until the maxtorque quantity will be less then specified threshold 
run_until_smaller('maxtorque', 1e-5 * gets('gamma') * 800e3)

# Set Gilbert damping constant to the realistic value
setv('alpha', 0.01)

# Set the initial time step for the solver, s
setv('dt', 1e-15)
# Reset the time to zero
setv('t', 0)

# Load slonczewski module
load('slonczewski')

# Define thickness of free layer pointwise, 0 value bypasses slonczewski
t_fl = makearray(1,Nx,Ny,Nz)
for m in range(Nx):
    for n in range(Ny):
        for o in range(Nz):
            t_fl[0][m][n][o] = 1.0
setmask('t_fl', t_fl)
setv('t_fl', sZ)

# Define labmda which controlls dot(m,p) see OOMMF manual
setv('lambda',1.0)
# Define polarization efficiency
setv('Pol',0.5669)
# Define prefactor of field-like torque, see OOMMF manual
setv('epsilon_prime', 0.0)

# Save initial magnetization to OVF2, PNG and VTK files.
save("m","ovf",[])
save("m","png",[])
save("m","vtk",[])

# Set polarization direction of the current, deg
pdeg = 1    
prad = pdeg * pi / 180.0
px = cos(prad)
py = sin(prad)

# Set total current, A
J = -0.008
carea = sX * sY
jc = J / carea  

print "Current density is: " +  str(jc) + "\n"
 
# Set current direction by using mask, uniform in this case
j=[ [[[0]]], [[[0]]], [[[1]]] ]
setmask('j', j)
setv('j', [0, 0, jc])
save("j","ovf",[])

# Set polarization by using mask, uniform in this case
p=[ [[[1]]], [[[1]]], [[[0]]] ]
setmask('p', p)
setv('p', [px, py, 0])
save("p","ovf",[])

# Save the magnetization to PNG files every 1 ps
#autosave("m", "png", [], 1e-12)
autosave("m", "gplot", [], 1e-12)
# Tabulate net magnetization every 1 ps, similar to OOMMF's mmGraph
autotabulate(["t", "<m>"], "m.txt", 1e-12)

# Run simulation for 5 ns
run(5e-9)
# Print execution statistics for debuging
printstats()
# Make sure everything is flushed!
sync()
