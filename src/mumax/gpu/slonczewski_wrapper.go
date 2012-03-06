//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for slonczewski_torque.cu
// Author: Graham Rowlands, Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func LLSlon(stt *Array, m *Array, p, alpha, Msat *Array,
	gamma float32, aj float32, bj float32, Pol float32, j *Array) {

	// Bookkeeping
	CheckSize(j.Size3D(), m.Size3D())
	CheckSize(p.Size3D(), m.Size3D())
	CheckSize(alpha.Size3D(), m.Size3D())
	CheckSize(Msat.Size3D(), m.Size3D())

	// Calling the CUDA functions
	C.slonczewski_async(
		(**C.float)(unsafe.Pointer(&(stt.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(p.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(p.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(p.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(alpha.Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(Msat.Pointers()[0]))),

		(C.float)(gamma),
		(C.float)(aj),
		(C.float)(bj),
		(C.float)(Pol),

		// The program X component is the user Z component!
		(**C.float)(unsafe.Pointer(&(j.Comp[X].Pointers()[0]))),

		(C.int)(m.PartLen3D()),
		(*C.CUstream)(unsafe.Pointer(&(stt.Stream[0]))))
}
