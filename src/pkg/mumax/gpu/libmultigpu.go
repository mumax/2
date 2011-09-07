//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


// This file wraps libmultigpu.so.
// Author: Arne Vansteenkiste

package gpu

//#include "libmultigpu.h"
import "C"

import (
		. "mumax/common"
	"unsafe"
)


// Adds 2 multi-GPU arrays: dst = a + b
func Add(dst, a, b *Array) {
	C.addAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}


func Torque(torque, m, h, alphaMap *Array, alphaMul float32){
	
	C.torqueAsync(
		(**C.float)(unsafe.Pointer(&(torque.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(torque.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(torque.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(alpha.Comp[Z].pointer[0]))),
		(C.float)(alphaMul),

		(C.int)(m.partLen))
	
}
