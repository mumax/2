//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for baryakhtar-longitudinal.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func BaryakhtarLongitudinalAsync(t *Array, h *Array, msat0T0 *Array, lambda *Array, lambdaMul []float64) {

	// Bookkeeping 
	CheckSize(h.Size3D(), t.Size3D())
	Assert(h.NComp() == 3)
    
	// Calling the CUDA functions
	C.baryakhtar_longitudinal_async(
		(**C.float)(unsafe.Pointer(&(t.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(msat0T0.Comp[X].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(lambda.Comp[XX].Pointers()[0]))), //XX
		(**C.float)(unsafe.Pointer(&(lambda.Comp[YY].Pointers()[0]))), //YY
		(**C.float)(unsafe.Pointer(&(lambda.Comp[ZZ].Pointers()[0]))), //ZZ
		(**C.float)(unsafe.Pointer(&(lambda.Comp[YZ].Pointers()[0]))), //YZ
		(**C.float)(unsafe.Pointer(&(lambda.Comp[XZ].Pointers()[0]))), //XZ
		(**C.float)(unsafe.Pointer(&(lambda.Comp[XY].Pointers()[0]))), //XY

		(C.float)(float32(lambdaMul[XX])),
		(C.float)(float32(lambdaMul[YY])),
		(C.float)(float32(lambdaMul[ZZ])),
		(C.float)(float32(lambdaMul[YZ])),
		(C.float)(float32(lambdaMul[XZ])),
		(C.float)(float32(lambdaMul[XY])),

		(*C.CUstream)(unsafe.Pointer(&(t.Stream[0]))),
		
		(C.int)(t.partLen3D))
}

