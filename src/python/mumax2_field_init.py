# This file contains function to set initial field given a array (with grid-size and cell-size already set up)


import os
import json
from mumax2 import *

##  Sets field (e.g. magnetization) as vortex (local or global) in the selected region.
## ARGUMENTS:
##		- fieldName (string) is the name of the field we should write to
##		- center (tuple 3 floats) is the center of the vortex
##		- axis (tuple 3 floats) is the direction of the core
##		- polarity (int) is the sense of the core (-1 for against the axis or +1 for along the axis)
##		- chirality (int) is the chirality of the vortex (-1 for CCW or +1 for CW)
##		- region (string) is the region name that should contain the vortex. None means all regions
## TODO:
##		- take into account Aex for setting the size of the core
##		- use gaussian function for the core
def setVortex( fieldName, center, axis, polarity, chirality, region = None, maxRadius = 0. ):
	## we assume that the engine return vectors as (z,y,x)
	## we assume that user will enter vector and center as (x,y,z)
	gridSize = getgridsize()
	cellSize = getcellsize()
	field = getArray(fieldName)
	regionDefinition = getArray('regionDefinition')
	axisNorm = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
	u = [axis[0] * axisNorm , axis[1] * axisNorm , axis[2] * axisNorm]
	for i, cell in enumerate(field):
		if not region or regionDefinition[i] == region:
			## coordinate of the current point
			coordinateZ = i / ( gridSize[1] * gridSize[2] )
			coordinateY = ( i - coordinateZ * ( gridSize[1] * gridSize[2] ) ) / gridSize[2]
			coordinateX = i % gridSize[2]
			coordinateZ *= cellSize[0]
			coordinateY *= cellSize[1]
			coordinateX *= cellSize[2]
			## component of v the shortest vector going from the line (passing by center and direction axis) and the current point
			v = center[0] - coordinateX, center[1] - coordinateY, center[2] - coordinateZ
			## scalar product v.u
			vScalaru = v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
			v[0] -= u[0] * vScalaru
			v[1] -= u[1] * vScalaru
			v[2] -= u[2] * vScalaru
			## v norm
			d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
			if not maxRadius or d <= maxRadius:
				## set field to vortex
				if d < 5e-9:
					field[i] = [ u[2] * polarity , u[1] * polarity , u[0] * polarity ]
				else:
					field[i] = - chirality * ( u[1] * v[2] - u[2] * v[1]) , - chirality * ( u[2] * v[0] - u[0] * v[2]) , - chirality * ( u[0] * v[1] - u[1] * v[0]) 
			
	return

## set up an array of the size of the grid filled with zeros and return it
## set up a dictionary of the regions name
def setupRegionSystem():# regionArrayName = 'region' ):
	regionDefinition = [ [[[0]]] ]
	setarray( 'regionDefinition', regionDefinition )
	global regionNameDictionary
	regionNameDictionary = {'empty':0}
	##setarray( regionArrayName, regionDefinition )
	return regionDefinition

## set up regions given a script that return a region index for each voxel
## ARGUMENTS:
##		- script is a function that will be called for each voxel. It should return a string to identify each region or 'empty'
##		  Its arguments should be three ints and a dictionary (x, y, z, parameters)
##		- parameters to pass to script 
def initRegions( script, parameters):
	global regionNameDictionary
	regionDefinition = getArray('regionDefinition')
	regionNameList = regionNameDictionary.values()
	regionNameListLen = len(regionNameList)
	for i, cell in enumerate(regionDefinition):
		## coordinate of the current point
		coordinateZ = i / ( gridSize[1] * gridSize[2] )
		coordinateY = ( i - coordinateZ * ( gridSize[1] * gridSize[2] ) ) / gridSize[2]
		coordinateX = i % gridSize[2]
		result = script(coordinateX, coordinateY, coordinateZ, parameters)
		regionDefinition[i] = regionNameDictionary.get(result,regionNameListLen)
		if regionDefinition[i] == regionNameListLen:
			regionNameDictionary[result] = regionNameListLen
			regionNameListLen +=1
	return