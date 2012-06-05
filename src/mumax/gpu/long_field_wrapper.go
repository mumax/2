//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for long_field.cu
// Author: Mykola Dvornik, Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)


func LongFieldAsync(hlf *Array, m *Array, msat *Array, msat0 *Array, kappa float64, msatMul float64, msat0Mul float64, stream Stream) {

	// Bookkeeping
	CheckSize(hlf.Size3D(), m.Size3D())
	CheckSize(msat0.Size3D(), m.Size3D())
	
	Assert(msat0.NComp()== 1)
    
	// Calling the CUDA functions
	C.long_field_async(
		(**C.float)(unsafe.Pointer(&(hlf.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(hlf.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(hlf.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(msat.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(msat0.Comp[X].Pointers()[0]))),
        
		(C.float)(kappa),
		(C.float)(msatMul),
		(C.float)(msat0Mul),
         
		(C.int)(m.PartLen3D()),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}
