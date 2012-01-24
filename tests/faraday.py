from mumax2 import *

# Test for Faraday's law

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('faraday')
save('kern_rotor', 'gplot', [], 'kern_rotor.gplot')
save('kern_rotor.xx', 'gplot', [], 'kern_rotorXX.gplot')
save('kern_rotor.xy', 'gplot', [], 'kern_rotorXY.gplot')
save('kern_rotor.xz', 'gplot', [], 'kern_rotorXZ.gplot')
save('kern_rotor.yx', 'gplot', [], 'kern_rotorYX.gplot')
save('kern_rotor.yy', 'gplot', [], 'kern_rotorYY.gplot')
save('kern_rotor.yz', 'gplot', [], 'kern_rotorYZ.gplot')
save('kern_rotor.zx', 'gplot', [], 'kern_rotorZX.gplot')
save('kern_rotor.zy', 'gplot', [], 'kern_rotorZY.gplot')
save('kern_rotor.zz', 'gplot', [], 'kern_rotorZZ.gplot')
save('kern_rotor', 'gplot', [], 'kern_rotor.gplot')
setpointwise('B_ext', 0, [0, 0, 0])
setpointwise('B_ext', 1e-9, [0, 0, 1]) 
setpointwise('B_ext', 2e-9, [0, 0, 0]) 

autotabulate(['B_ext', '<B>', '<dB_dt>'], 'B.txt', 1e-12)
setv('dt', 1e-12)
run(3e-9)

printstats()
savegraph("graph.png")
