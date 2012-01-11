//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file wraps libmultigpu.so.
// Author: Arne Vansteenkiste

package gpu

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

// Initialise scalar quantity with uniform value in each region
func InitScalarQuantUniformRegion(S, regions *Array, initValues []float32) {
	C.initScalarQuantUniformRegionAsync(
		(**C.float)(unsafe.Pointer(&(S.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(regions.pointer[0]))),
		(*C.float)(unsafe.Pointer(&(initValues[0]))),
		(C.int)(len(initValues)),

		(*C.CUstream)(unsafe.Pointer(&(S.Stream[0]))),

		(C.int)(regions.partLen3D))
	S.Stream.Sync()
}

// Initialise scalar quantity with uniform value in each region
func InitVectorQuantUniformRegion(S, regions *Array, initValuesX, initValuesY, initValuesZ []float32) {
	C.initVectorQuantUniformRegionAsync(
		(**C.float)(unsafe.Pointer(&(S.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(regions.pointer[0]))),

		(*C.float)(unsafe.Pointer(&(initValuesX[0]))),
		(*C.float)(unsafe.Pointer(&(initValuesY[0]))),
		(*C.float)(unsafe.Pointer(&(initValuesZ[0]))),

		(C.int)(len(initValuesX)),

		(*C.CUstream)(unsafe.Pointer(&(S.Stream[0]))),

		(C.int)(regions.partLen3D))
	S.Stream.Sync()
}

// Initialise scalar quantity with uniform value in each region
func InitVectorQuantVortexRegion(S, regions *Array, regionsToProceed []bool, center, axis, cellsize []float32, polarity, chirality int, maxRadius float32) {
	C.initVectorQuantVortexRegionAsync(
		(**C.float)(unsafe.Pointer(&(S.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[X].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(regions.pointer[0]))),

		(*C.bool)(unsafe.Pointer(&(regionsToProceed[0]))),
		
		(C.float)(center[0]),
		(C.float)(center[1]),
		(C.float)(center[2]),
		
		(C.float)(axis[0]),
		(C.float)(axis[1]),
		(C.float)(axis[2]),
		
		(C.float)(cellsize[0]),
		(C.float)(cellsize[1]),
		(C.float)(cellsize[2]),

		(C.int)(polarity),
		(C.int)(chirality),
		
		(C.float)(maxRadius),

		(C.int)(len(regionsToProceed)),

		(*C.CUstream)(unsafe.Pointer(&(S.Stream[0]))),

		(C.int)(regions.partLen3D),
		
		(C.int)(regions.partSize[2]),
		(C.int)(regions.partSize[1]),
		(C.int)(regions.partSize[0]))
	S.Stream.Sync()
}