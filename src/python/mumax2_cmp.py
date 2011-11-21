# This file contains function to set initial field given a array (with grid-size and cell-size already set up)


import os
import json
import png
import re
import sys
import math
from mumax2 import *

##  Sets field (e.g. magnetization) as vortex (local or global) in the selected region.
## ARGUMENTS:
##		- fieldName (string) is the name of the field we should write to
##		- center (tuple 3 floats) is the center of the vortex
##		- axis (tuple 3 floats) is the direction of the core
##		- polarity (int) is the sense of the core (-1 for against the axis or +1 for along the axis)
##		- chirality (int) is the chirality of the vortex (-1 for CCW or +1 for CW)
##		- region (string) is the region name that should contain the vortex. 'all' means all regions
## TODO:
##		- take into account Aex for setting the size of the core
##		- use gaussian function for the core
def setVortex( fieldName, center, axis, polarity, chirality, region = 'all', maxRadius = 0. ):
	## we assume that the engine return vectors as (z,y,x)
	## we assume that user will enter vector and center as (x,y,z)
	gridSize = getgridsize()
	cellSize = getcellsize()
	field = getarray(fieldName)
	setupRegionSystem()
	regionDefinition = getarray('regionDefinition')
	global regionNameDictionary
	axisNorm = math.sqrt(1/(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]))
	u = [axis[0] * axisNorm , axis[1] * axisNorm , axis[2] * axisNorm]
	for X in range(0,gridSize[0]):
		for Y in range(0,gridSize[1]):
			for Z in range(0,gridSize[2]):
			#for i, cell in enumerate(field):
				## if all region selected and not in empty, or if in selected region
				#print >> sys.stderr, 'i =%d' % i
				## coordinate of the current p	oint
				#Z = i / ( gridSize[1] * gridSize[2] )
				#Y = ( i - Z * ( gridSize[1] * gridSize[2] ) ) / gridSize[2]
				#X = i % gridSize[2]	
				cellRegion = getcell('regionDefinition', X,Y,Z)
				#if ( region == 'all' and cellRegion != 0. ) or cellRegion == region:
				coordinateZ = float(Z) * cellSize[2]
				coordinateY = float(Y) * cellSize[1]
				coordinateX = float(X) * cellSize[0]	
				## component of v the shortest vector going from the line (passing by center and direction axis) and the current point
				v1 = [center[0] - coordinateX, center[1] - coordinateY, center[2] - coordinateZ]
				## scalar product v.u
				vScalaru = v1[0] * u[0] + v1[1] * u[1] + v1[2] * u[2]
				v = [0.,0.,0.]
				v[0] = v1[0] - u[0] * vScalaru
				v[1] = v1[1] - u[1] * vScalaru
				v[2] = v1[2] - u[2] * vScalaru
				## v norm
				d = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
				#print >> sys.stderr, '%d\t%d\t%d :\t%g\t%g\t:\t%g\t%g\t%g\t:\t%g\t%g\t%g' % (X, Y, Z, vScalaru, d, v1[0], v1[1], v1[2], v[0], v[1], v[2])
				if maxRadius == 0. or d <= maxRadius:
					print >> sys.stderr, 'set M' 
					#Ms = getcell('Msat',X,Y,Z)
					## set field to vortex
					if d < 2e-8:
						m = [ u[0] * polarity ,
							  u[1] * polarity ,
							  u[2] * polarity ]
						setcell(fieldName,X,Y,Z,m)
						m = getcell('m',X,Y,Z)
						print >> sys.stderr, 'core :\t%d\t%d\t%d :\t%g\t:\t%g\t%g\t%g' % (X, Y, Z, d, m[0], m[1], m[2])
					else:
						m = [ - chirality * ( u[1] * v[2] - u[2] * v[1])/d ,
							  - chirality * ( u[2] * v[0] - u[0] * v[2])/d ,
							  - chirality * ( u[0] * v[1] - u[1] * v[0])/d ]
						print >> sys.stderr, 'out :\t%g\t%g\t%g :\t%g\t:\t%g\t%g\t%g' % (coordinateX, coordinateY, coordinateZ, d, m[0], m[1], m[2])
						setcell(fieldName,X,Y,Z,[float(m[0]),float(m[1]),float(m[2])])
						m = getcell('m',X,Y,Z)
						print >> sys.stderr, 'out :\t%g\t%g\t%g :\t%g\t:\t%g\t%g\t%g' % (coordinateX, coordinateY, coordinateZ, d, m[0], m[1], m[2])
	return

## set up an array of the size of the grid filled with zeros and return it
## set up a dictionary of the regions name
def setupRegionSystem():# regionArrayName = 'region' ):
	tmp = [ [[[0]]] ]
	setscalar('regionDefinition', 1.)
	#setmask( 'regionDefinition', tmp )
	global regionNameDictionary
	regionNameDictionary = {'empty':0}
	##setarray( regionArrayName, regionDefinition )
	return

## set up regions given a script that return a region index for each voxel
## ARGUMENTS:
##		- script is a function that will be called for each voxel. It should return a string to identify each region or 'empty'
##		  Its arguments should be three ints and a dictionary (x, y, z, parameters)
##		- parameters to pass to script 
def initRegions( script, parameters):
	global regionNameDictionary
	regionDefinition = getarray('regionDefinition')
	regionNameList = regionNameDictionary.values()
	regionNameListLen = float(len(regionNameList))
	for i, cell in enumerate(regionDefinition):
		## coordinate of the current point
		Z = i / ( gridSize[1] * gridSize[2] )
		Y = ( i - Z * ( gridSize[1] * gridSize[2] ) ) / gridSize[2]
		X = i % gridSize[2]
		result = script(X, Y, Z, parameters)
		tmp = regionNameDictionary.get(result,regionNameListLen)
		setcell('regionDefinition',X,Y,Z,[ tmp ])
		if tmp == regionNameListLen:
			regionNameDictionary[result] = regionNameListLen
			regionNameListLen += 1.
		del tmp
	return

## set up regions given q png imqge by extruding it perpendicularly to the plane given in variable plane
## ARGUMENTS:
##		- imageName (string) is the name of the PNG image to use
##		- regionList (dictionary string=>string) associate a color to each region.
##		  The color could be either named if it is part of the standart html color (see http://www.w3schools.com/html/html_colornames.asp)
##		  or a string coding the color in the HTML hexadecimal format : '#XXXXXX' where X are between 0 and F
##		- plane (string) defines the plane to which the picture will be applied. By default the plan xy
##		  the first axis will be matched with the width of the image and the second axis with the height
##		- thickness (float) defines the extruded thickness. 0 means across the whole volume.
##		- origin (float) if thickness is not 0, then origin defines the starting point of the extrusion.
##		  It will happen along the increasing value of the extrusion axis.
def extrudeImage( imageName, regionList, plane = 'xy'):#, thickness = 0., origin = 0 ):
	global regionNameDictionary
	#test plane argument validity
	planeValidity = re.compile('[x-z]{2}',re.IGNORECASE)
	if not re.match(regionList[i]) or plane[0] == plane[1]:
		print >> sys.stderr, 'extrudeImage plane cannot be %s' % plane
		sys.exit()
	htmlColorName = {
					'AliceBlue'		: '#F0F8FF',				'AntiqueWhite'	: '#FAEBD7',
					'Aqua'			: '#00FFFF',				'Aquamarine'	: '#7FFFD4',
					'Azure'			: '#F0FFFF',				'Beige'			: '#F5F5DC',
					'Bisque'		: '#FFE4C4',				'Black'			: '#000000',
					'BlanchedAlmond': '#FFEBCD',				'Blue'			: '#0000FF',
					'BlueViolet'	: '#8A2BE2',				'Brown'			: '#A52A2A',
					'BurlyWood'		: '#DEB887',				'CadetBlue'		: '#5F9EA0',
					'Chartreuse'	: '#7FFF00',				'Chocolate'		: '#D2691E',
					'Coral'			: '#FF7F50',				'CornflowerBlue': '#6495ED',
					'Cornsilk'		: '#FFF8DC',				'Crimson'		: '#DC143C',
					'Cyan'			: '#00FFFF',				'DarkBlue'		: '#00008B',
					'DarkCyan'		: '#008B8B',				'DarkGoldenRod'	: '#B8860B',
					'DarkGray'		: '#A9A9A9',				'DarkGrey'		: '#A9A9A9',
					'DarkGreen'		: '#006400',				'DarkKhaki'		: '#BDB76B',
					'DarkMagenta'	: '#8B008B',				'DarkOliveGreen': '#556B2F',
					'Darkorange'	: '#FF8C00',				'DarkOrchid'	: '#9932CC',
					'DarkRed'		: '#8B0000',				'DarkSalmon'	: '#E9967A',
					'DarkSeaGreen'	: '#8FBC8F',				'DarkSlateBlue'	: '#483D8B',
					'DarkSlateGray'	: '#2F4F4F',				'DarkSlateGrey'	: '#2F4F4F',
					'DarkTurquoise'	: '#00CED1',				'DarkViolet'	: '#9400D3',
					'DeepPink'		: '#FF1493',				'DeepSkyBlue'	: '#00BFFF',
					'DimGray'		: '#696969',				'DimGrey'		: '#696969',
					'DodgerBlue'	: '#1E90FF',				'FireBrick'		: '#B22222',
					'FloralWhite'	: '#FFFAF0',				'ForestGreen'	: '#228B22',
					'Fuchsia'		: '#FF00FF',				'Gainsboro'		: '#DCDCDC',
					'GhostWhite'	: '#F8F8FF',				'Gold'			: '#FFD700',
					'GoldenRod'		: '#DAA520',				'Gray'			: '#808080',
					'Grey'			: '#808080',				'Green'			: '#008000',
					'GreenYellow'	: '#ADFF2F',				'HoneyDew'		: '#F0FFF0',
					'HotPink'		: '#FF69B4',				'IndianRed'		: '#CD5C5C',
					'Indigo'		: '#4B0082',				'Ivory'			: '#FFFFF0',
					'Khaki'			: '#F0E68C',				'Lavender'		: '#E6E6FA',
					'LavenderBlush'	: '#FFF0F5',				'LawnGreen'		: '#7CFC00',
					'LemonChiffon'	: '#FFFACD',				'LightBlue'		: '#ADD8E6',
					'LightCoral'	: '#F08080',				'LightCyan'		: '#E0FFFF',
					'LightGoldenRodYellow'	: '#FAFAD2',		'LightGray'		: '#D3D3D3',
					'LightGrey'		: '#D3D3D3',				'LightGreen'	: '#90EE90',
					'LightPink'		: '#FFB6C1',				'LightSalmon'	: '#FFA07A',
					'LightSeaGreen'	: '#20B2AA',				'LightSkyBlue'	: '#87CEFA',
					'LightSlateGray': '#778899',				'LightSlateGrey': '#778899',
					'LightSteelBlue': '#B0C4DE',				'LightYellow'	: '#FFFFE0',
					'Lime'			: '#00FF00',				'LimeGreen'		: '#32CD32',
					'Linen'			: '#FAF0E6',				'Magenta'		: '#FF00FF',
					'Maroon'		: '#800000',				'MediumAquaMarine'	: '#66CDAA',
					'MediumBlue'	: '#0000CD',				'MediumOrchid'	: '#BA55D3',
					'MediumPurple'	: '#9370D8',				'MediumSeaGreen'	: '#3CB371',
					'MediumSlateBlue'	: '#7B68EE',			'MediumSpringGreen'	: '#00FA9A',
					'MediumTurquoise'	: '#48D1CC',			'MediumVioletRed'	: '#C71585',
					'MidnightBlue'	: '#191970',				'MintCream'		: '#F5FFFA',
					'MistyRose'		: '#FFE4E1',				'Moccasin'		: '#FFE4B5',
					'NavajoWhite'	: '#FFDEAD',				'Navy'			: '#000080',
					'OldLace'		: '#FDF5E6',				'Olive'			: '#808000',
					'OliveDrab'		: '#6B8E23',				'Orange'		: '#FFA500',
					'OrangeRed'		: '#FF4500',				'Orchid'		: '#DA70D6',
					'PaleGoldenRod'	: '#EEE8AA',				'PaleGreen'		: '#98FB98',
					'PaleTurquoise'	: '#AFEEEE',				'PaleVioletRed'	: '#D87093',
					'PapayaWhip'	: '#FFEFD5',				'PeachPuff'		: '#FFDAB9',
					'Peru'			: '#CD853F',				'Pink'			: '#FFC0CB',
					'Plum'			: '#DDA0DD',				'PowderBlue'	: '#B0E0E6',
					'Purple'		: '#800080',				'Red'			: '#FF0000',
					'RosyBrown'		: '#BC8F8F',				'RoyalBlue'		: '#4169E1',
					'SaddleBrown'	: '#8B4513',				'Salmon'		: '#FA8072',
					'SandyBrown'	: '#F4A460',				'SeaGreen'		: '#2E8B57',
					'SeaShell'		: '#FFF5EE',				'Sienna'		: '#A0522D',
					'Silver'		: '#C0C0C0',				'SkyBlue'		: '#87CEEB',
					'SlateBlue'		: '#6A5ACD',				'SlateGray'		: '#708090',
					'SlateGrey'		: '#708090',				'Snow'			: '#FFFAFA',
					'SpringGreen'	: '#00FF7F',				'SteelBlue'		: '#4682B4',
					'Tan'			: '#D2B48C',				'Teal'			: '#008080',
					'Thistle'		: '#D8BFD8',				'Tomato'		: '#FF6347',
					'Turquoise'		: '#40E0D0',				'Violet'		: '#EE82EE',
					'Wheat'			: '#F5DEB3',				'White'			: '#FFFFFF',
					'WhiteSmoke'	: '#F5F5F5',				'Yellow'		: '#FFFF00',
					'YellowGreen'	: '#9ACD32'
					}
	#first convert regionList to be fully coded in color hex code and fill regionNameDictionanry
	htmlCodeRE = re.compile('\#[a-f\d]{6}',re.IGNORECASE)
	colorToHexCode = {}
	regionNameListLen = float(len(regionNameDictionary))
	for i, cell in enumerate(regionList):
		if cell[0] != '#':
			regionList[i] = htmlColorName[cell]
		if re.match(regionList[i]):
			regionNameDictionary[i] = regionNameListLen
			colorToHexCode[cell] = regionNameListLen
			regionNameListLen += 1.
	setupRegionSystem()
	#Read picture
	imageReader = png.Reader( filename = imageName )
	imageWidth, imageHeight, pngData, _ = png.read()
	rawRegionData = [[0.]]
	for rowIndex, row in enumerate(pngdata):
		for columnIndex, pixel in enumerate(row):
			colorCode = "#%02X%02X%02X" % (pixel[0],pixel[1],pixel[2])
			if colorCode in colorToHexCode:
				rawRegionData[rowIndex][columnIndex] = colorToHexCode[colorCode]
			else:
				rawRegionData[rowIndex][columnIndex] = regionNameDictionary['empty']
	
	#Apply rawRegionData to regionDefinition mask by stretching in plane and extruding out of plane
	#rawRegionData
	gridSize = getgridsize()
	u = 0
	v = 0
	w = 0
	if plane[0] == 'x':
		u=0
	elif plane[0] == 'y':
		u=1
	elif plane[0] == 'z':
		u=2
	if plane[1] == 'x':
		v=0
	elif plane[1] == 'y':
		v=1
	elif plane[1] == 'z':
		v=2
	if u+v == 1:
		w = 2
	elif u+v == 2:
		w = 1
	else:
		w = 0
	stretchedRegionData = [[0.]]
	for i in range(0,gridSize[u]):
		i1 = (i * gridSize[u]) / imageWidth
		for j in range(0,gridSize[v]):
			j1 = (j * gridSize[v]) / imageHeight
			for k in range(0,gridSize[w]):
				setCell('regionDefinition',i,j,k,[rawRegionData[i1][j1]])
