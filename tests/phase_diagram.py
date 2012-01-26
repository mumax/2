from mumax2 import *
from mumax2_cmp import *
from mumax2_geom import *

########################################################################################
##                                     Parameters                                     ##
########################################################################################
radius = 300e-9                   # radius of the polygon (m)
thickness = 4e-9                  # thickness of the polygon (m)
insulatorThickness = 4e-9         # thickness of the insulator between both polygons (m)
sides = 5                         # number of sides of the polygon (unitless)
rotation = 0                      # angle of rotation of the polygon (degre)
centerX = radius                  # X coordinate of the center of the polygon (m)
centerY = radius                  # Y coordinate of the center of the polygon (m)
centerZ = thickness/2             # Z coordinate of the center of the polygon (m)
Nx = 128                          # number of cell in x direction (unitless)
Ny = 128                          # number of cell in y direction (unitless)
Nz = 5                            # number of cell in z direction (unitless)
Sx = 2 * radius                   # total size of universe in x direction (m)
Sy = 2 * radius                   # total size of universe in y direction (m)
Sz = 3*thickness                # total size of universe in z direction (m)

########################################################################################
##                                   End parameters                                   ##
########################################################################################

setgridsize(Nx, Ny, Nz)
setcellsize(Sx/Nx, Sy/Ny, Sz/Nz)


load('regions')

Ngone(sides, radius, rotation, centerX, centerY, 0, thickness, "pentagon1")
Ngone(sides, radius, rotation, centerX, centerY, thickness+insulatorThickness, 2*thickness+insulatorThickness, "pentagon2")
save("regionDefinition", "ovf", ["Text"], "phase_diagram_region.ovf" )

load('micromagnetism')
load('solver/rk12')

setv('dt', 1e-13)
setv('m_maxerror', 1./1000)
setv('Aex', 1.3e-11)
setv('alpha', 0.1)
setv('Msat', 1)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

MsatValues = {"pentagon1":800e3,
              "pentagon2":400e3}
InitUniformRegionScalarQuant('Msat', MsatValues)
#InitRandomUniformRegionScalarQuant('Msat', mValues, 800e3, 400e3 )


#InitRandomUniformRegionVectorQuant('m', mValues)
mValues = {"pentagon1":1.0}
InitVortexRegionVectorQuant('m', mValues, [radius,radius,0.0], [0.0,0.0,1.0], 1, 1, 0 )
mValues = {"pentagon2":1.0}
InitVortexRegionVectorQuant('m', mValues, [radius,radius,0.0], [0.0,0.0,1.0], -1, -1, 0 )
save("m", "ovf", ["Text"], "phase_diagram_m.ovf" )
save("Msat", "ovf", ["Text"], "phase_diagram_Msat.ovf" )
#setarray('m', regionDefinition)

#run(2e-9)
step()
save("m", "ovf", ["Text"], "phase_diagram_m_end.ovf" )

printstats()

savegraph("graph.png")