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
	cu "cuda/driver"
	"unsafe"
)

// Initialise scalar quantity with uniform value in each region
func InitScalarQuantUniformRegion(S, regions *Array, initValues []float64) {
	//initialValues = make([]cu.DevicePtr, 1)
	initialValues := cu.MemAlloc(SIZEOF_FLOAT * int64(len(initValues)))
	/*cu.MemcpyHtoD(cu.DevicePtr(offset(uintptr(initialValues), SIZEOF_FLOAT * len(initValues))),
	cu.HostPtr(unsafe.Pointer(&(initValues))),
	int64(len(initValues))*SIZEOF_FLOAT)*/
	C.initScalarQuantUniformRegionAsync(
		(**C.float)(unsafe.Pointer(&(S.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(regions.pointer[0]))),
		(*C.float)(unsafe.Pointer(initialValues)),
		(C.int)(len(initValues)),

		(*C.CUstream)(unsafe.Pointer(&(S.Stream[0]))),

		(C.int)(regions.partLen3D))
	S.Stream.Sync()
	initialValues.Free()

}

// Initialise scalar quantity with uniform value in each region
/*func initVectorQuantUniformRegion(S, regions *Array, initValues [][]float64) {
	C.initVectorQuantUniformRegionAsync(
		(**C.float)(unsafe.Pointer(&(S.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(regions.pointer[0]))),

		(*C.float)(unsafe.Pointer(initValues[0])),
		(*C.float)(unsafe.Pointer(initValues[1])),
		(*C.float)(unsafe.Pointer(initValues[2])),

		(*C.CUstream)(unsafe.Pointer(&(S.Stream[0]))),

		(C.int)(regions.partLen3D))
	S.Stream.Sync()

}
*/
