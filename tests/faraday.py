from mumax2 import *

# Test for Faraday's law

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
Cx = 5e-9
Cy = 5e-9
Cz = 5e-9
setcellsize(Cx, Cy, Cz)

load('faraday')
savegraph("graph.png")
#save('kern_rotor', 'gplot', [], 'kern_rotor.gplot')
#save('kern_rotor.xx', 'gplot', [], 'kern_rotorXX.gplot')
#save('kern_rotor.xy', 'gplot', [], 'kern_rotorXY.gplot')
#save('kern_rotor.xz', 'gplot', [], 'kern_rotorXZ.gplot')
#save('kern_rotor.yx', 'gplot', [], 'kern_rotorYX.gplot')
#save('kern_rotor.yy', 'gplot', [], 'kern_rotorYY.gplot')
#save('kern_rotor.yz', 'gplot', [], 'kern_rotorYZ.gplot')
#save('kern_rotor.zx', 'gplot', [], 'kern_rotorZX.gplot')
#save('kern_rotor.zy', 'gplot', [], 'kern_rotorZY.gplot')
#save('kern_rotor.zz', 'gplot', [], 'kern_rotorZZ.gplot')
#save('kern_rotor', 'gplot', [], 'kern_rotor.gplot')
setpointwise('B_ext', 0, [0, 0, 0])
setpointwise('B_ext', 1e-9, [0, 0, 1]) 
setpointwise('B_ext', 2e-9, [0, 0, 0]) 

autotabulate(['t', 'B_ext', '<B>', '<dB_dt>'], 'B.txt', 1e-12)
autosave('E', 'omf', ['Text'], 1e-12)
autosave('E', 'gplot', [], 1e-12)
setv('dt', 1e-12)
run(100e-12)

E=getarray('E')

x0 = 3
y0 = 9
x1 = 8
y1 = 12

emf = 0
for i in range(x0,x1):
	j = y0
	emf += E[0][i][j][0]

i = x1
for j in range(y0,y1):
	emf += E[1][i][j][0]

j=y1
for i in range(x0,x1):
	j = y0
	emf -= E[0][i][j][0]

i=x0
for j in range(y0,y1):
	emf -= E[1][i][j][0]

S = (x1-x0-1)*Cx * (y1-y0)*Cy
dBdt=1e9
want = S * dBdt
echo("emf " + str(emf) + " want: " + str(want))
dB = getv("<dB_dt>")
echo("<dBdt> " + str(dB))

