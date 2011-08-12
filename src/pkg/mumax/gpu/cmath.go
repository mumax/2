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

// Adds 2 multi-GPU arrays: dst = a + b
func CAdd(dst, a, b *Array) {
	C.add(
		(**C.float)(unsafe.Pointer(&(dst.devPtr[0]))),
		(**C.float)(unsafe.Pointer(&(a.devPtr[0]))),
		(**C.float)(unsafe.Pointer(&(b.devPtr[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.devStream[0]))),
		C.int(NDevice()),
		C.int(dst.partLength4D))
}
