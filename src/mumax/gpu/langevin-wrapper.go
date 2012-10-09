//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for langevin.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func LangevinAsync(msat0 *Array, msat0T0 *Array, T *Array, J0 *Array, msat0Mul float64, msat0T0Mul float64, J0Mul float64, stream Stream) {

	// Bookkeeping
	CheckSize(msat0.Size3D(), T.Size3D())

	// Calling the CUDA functions
	C.langevinAsync(
		(**C.float)(unsafe.Pointer(&(msat0.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(msat0T0.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(T.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(J0.Comp[X].Pointers()[0]))),
        
		(C.float)(msat0Mul),
		(C.float)(msat0T0Mul),
		(C.float)(J0Mul),

		(C.int)(msat0.partLen3D),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}
