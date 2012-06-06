from mumax2 import *
from mumax2_cmp import *

#regionNameDictionary ={}

setgridsize(256, 128, 2)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('solver/rk12')

setv('alpha', 0.1)
getscalar('alpha')
setv('Msat', 1.0)
setv('Aex', 12e-13)
setv('m_maxerror', 1./1000)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

## Example of initialization of region system with a picture
imageName = os.path.abspath( os.path.dirname(__file__)) + '/engine_cmp_regions.png'
regionDic = {"M":"Blue",
			 "u":"Lime",
			 "m":"Black",
			 "a":"DarkBlue",
			 "x":"Red",
			 "2":"Yellow"}
extrudeImage( imageName, regionDic)
MsatValues = {"M":1.e6,
			  "u":2.e6,
			  "m":3.e6,
			  "a":4.e6,
			  "x":5.e6,
			  "2":6.e6}
InitUniformRegionScalarQuant('Msat', MsatValues)

mValues = {"M":[ 1.0, 0.0,0.0],
		   "u":[ 1.0, 1.0,0.0],
		   "m":[ 0.0, 1.0,0.0],
		   "a":[-1.0, 1.0,0.0],
		   "x":[-1.0, 0.0,0.0],
		   "2":[-1.0,-1.0,0.0]}
InitUniformRegionVectorQuant('m', mValues)
mValues = {"M":0,
		   "u":0,
		   "m":0,
		   "a":1,
		   "x":0,
		   "2":0}
#InitVortexRegionVectorQuant('m', mValues, [200.0e-9,200.0e-9,0.0], [0.0,0.0,1.0], 1, 1, 0 )
InitVortexRegionVectorQuant('m', mValues, [776.0e-9,213.0e-9,0.0], [0.0,0.0,1.0], 1, 1, 0 )

save("Msat", "ovf", ["Text"], "regions_picture_Ms.ovf" )
save("m", "ovf", ["Text"], "regions_picture_uniform_m.ovf" )

## Example of initialization of region system with a script
def script(X, Y, Z, param):
	if Y > X * param["slope"]:
		return 'Upper'
	else:
		return 'Lower'
gridSize = getgridsize()
parameters = {"slope" : gridSize[1]/gridSize[0]}
#initRegionsScript( script , parameters)
mValues = {"Upper":1e6,
		   "Lower":2e6}
#InitUniformRegionScalarQuant('Msat', mValues)
#save("Msat", "ovf", ["Text"], "regions_script.ovf" )

Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 

setv('H_ext', [Hx, Hy, Hz])

setv('dt', 1e-12)
#save("regionDefinition", "omf", ["Text"], "region.omf" )
#autosave("m", "omf", ["Text"], 10e-12)
#autosave("m", "ovf", ["Text"], 10e-12)
#autosave("m", "bin", [], 10e-12)
#autotabulate(["t", "H_ext"], "t.txt", 10e-12)
for i in range(100):
	step()

printstats()
savegraph("graph.png")





