//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


package gpu

//#include "libmumax2.h"
import "C"

import (
	//. "mumax/common"
	"unsafe"
)

// INTERNAL
const MAXGPU = 4

// Adds 2 multi-GPU arrays: dst = a + b
func CAdd(dst, a, b *Array) {
	// avoid allocation:
	var dstPtr [MAXGPU]unsafe.Pointer
	var aPtr [MAXGPU]unsafe.Pointer
	var bPtr [MAXGPU]unsafe.Pointer
	var devStream [MAXGPU]unsafe.Pointer

	for i := range dst.pointer {
		dstPtr[i] = unsafe.Pointer(dst.pointer[i])
		aPtr[i] = unsafe.Pointer(a.pointer[i])
		bPtr[i] = unsafe.Pointer(b.pointer[i])
		devStream[i] = unsafe.Pointer(dst.stream[i])
	}
	partLength4D := dst.partLen4D

	C.add(
		(**C.float)(unsafe.Pointer(&(dstPtr[0]))),
		(**C.float)(unsafe.Pointer(&(aPtr[0]))),
		(**C.float)(unsafe.Pointer(&(bPtr[0]))),
		(*C.CUstream)(unsafe.Pointer(&(devStream[0]))),
		C.int(NDevice()),
		C.int(partLength4D))
}
