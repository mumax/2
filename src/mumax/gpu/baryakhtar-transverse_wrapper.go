//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for baryakhtar-transverse.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func BaryakhtarTransverseAsync(t *Array, M *Array, h *Array, msat0T0 *Array, mu *Array, muMul []float64) {

	// Bookkeeping 
	CheckSize(h.Size3D(), M.Size3D())
	CheckSize(h.Size3D(), t.Size3D())
	Assert(h.NComp() == 3)
    
	// Calling the CUDA functions
	C.baryakhtar_transverse_async(
		(**C.float)(unsafe.Pointer(&(t.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(M.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(M.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(M.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(msat0T0.Comp[X].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(mu.Comp[XX].Pointers()[0]))), //XX
		(**C.float)(unsafe.Pointer(&(mu.Comp[YY].Pointers()[0]))), //YY
		(**C.float)(unsafe.Pointer(&(mu.Comp[ZZ].Pointers()[0]))), //ZZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[YZ].Pointers()[0]))), //YZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[XZ].Pointers()[0]))), //XZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[XY].Pointers()[0]))), //XY

		(C.float)(float32(muMul[XX])),
		(C.float)(float32(muMul[YY])),
		(C.float)(float32(muMul[ZZ])),
		(C.float)(float32(muMul[YZ])),
		(C.float)(float32(muMul[XZ])),
		(C.float)(float32(muMul[XY])),

		(*C.CUstream)(unsafe.Pointer(&(t.Stream[0]))),
		(C.int)(t.partLen3D))
}

