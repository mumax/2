//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for kappa.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func EnergyFlowAsync(w *Array, mf *Array, R *Array, Tc *Array, S *Array, n *Array, SMul float64, stream Stream) {

	// Calling the CUDA functions
	C.energyFlowAsync(
		(**C.float)(unsafe.Pointer(&(w.Comp[X].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(mf.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(mf.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(mf.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(R.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(R.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(R.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(Tc.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(S.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(n.Comp[X].Pointers()[0]))),

		(C.float)(SMul),

		(C.int)(w.partLen3D),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}
