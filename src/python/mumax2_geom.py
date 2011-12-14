## @package mumax2_geom
# Functions to generate masks representing geometrical shapes
# @author Arne Vansteenkiste

from mumax2 import *

## Returns an array with ones inside an ellipsoid with semi-axes rx,ry,rz
#  and 0 outside. Typically used as a mask.
def ellipsoid(rx, ry, rz):
	Nx,Ny,Nz = getgridsize()
	Cx,Cy,Cz = getcellsize()
	mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	for i in range(0,Nx):
			x = ((-Nx/2. + i + 0.5) * Cx)
			for j in range(0,Ny):
				y = ((-Ny/2. + j + 0.5) * Cy)
				for k in range(0,Nz):
					z = ((-Nz/2. + k + 0.5) * Cz)
					if (x/rx)**2 + (y/ry)**2 + (z/rz)**2 < 1:
							mask[i][j][k] = 1
					else:
							mask[i][j][k] = 0
	print "mask", mask
	return [mask]



def ellipse():
	Nx,Ny,Nz = getgridsize()
	mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	rx = Nx/2.
	ry = Ny/2.
	for i in range(0,Nx):
			x = (-Nx/2. + i + 0.5)
			for j in range(0,Ny):
				y = (-Ny/2. + j + 0.5)
				for k in range(0,Nz):
					if (x/rx)**2 + (y/ry)**2 < 1:
							mask[i][j][k] = 1
					else:
							mask[i][j][k] = 0
	echo("mask"+  str(mask))
	return [mask]



