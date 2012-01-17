from mumax2 import *
from mumax2_cmp import *
from mumax2_geom import *

NgoneFit(5, 300e-9, 30, 128, 128, 2, 2e-9, "pentagon")
save("regionDefinition", "ovf", ["Text"], "phase_diagram_region.ovf" )

load('micromagnetism')

#setv('m_maxerror', 1./1000)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('Msat', 1)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

alphaValues = {"pentagon":0.1}
InitUniformRegionScalarQuant('alpha', alphaValues)
MsatValues = {"pentagon":800e3}
#InitUniformRegionScalarQuant('Msat', MsatValues)
mValues = {"pentagon":1.0}
InitRandomUniformRegionScalarQuant('Msat', mValues, 800e3, 400e3 )
AexValues = {"pentagon":1.3e-11}
#InitUniformRegionScalarQuant('Aex', AexValues)
save("Msat", "ovf", ["Text"], "phase_diagram_Msat.ovf" )


InitRandomUniformRegionVectorQuant('m', mValues)
#InitVortexRegionVectorQuant('m', mValues, [150.0e-9,150.0e-9,0.0], [0.0,0.0,1.0], 1, 1, 0 )
save("m", "ovf", ["Text"], "phase_diagram_m.ovf" )
#setarray('m', regionDefinition)
