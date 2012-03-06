//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for zhang-li_torque.cu
// Author: Rémy Lassalle-Balier

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func LLZhang(m, h, j, alpha, bj, cj, Msat *Array, ux, uy, uz, dt_gilbert float32) {
	CheckSize(h.Size3D(), m.Size3D())
	CheckSize(j.Size3D(), m.Size3D())
	CheckSize(alpha.Size3D(), m.Size3D())
	CheckSize(bj.Size3D(), m.Size3D())
	CheckSize(cj.Size3D(), m.Size3D())
	CheckSize(Msat.Size3D(), m.Size3D())
	C.spintorque_deltaMAsync(
		(**C.float)(unsafe.Pointer(&(m.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].Pointers()[1]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].Pointers()[2]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].Pointers()[1]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].Pointers()[2]))),

		(**C.float)(unsafe.Pointer(&(alpha.Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(bj.Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(cj.Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(Msat.Pointers()[0]))),

		(C.float)(ux),
		(C.float)(uy),
		(C.float)(uz),

		(**C.float)(unsafe.Pointer(&(j.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Y].Pointers()[1]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Z].Pointers()[2]))),

		(C.float)(dt_gilbert),

		(*C.CUstream)(unsafe.Pointer(&(m.Stream[0]))),

		(C.int)(m.PartSize()[X]),
		(C.int)(m.PartSize()[Y]),
		(C.int)(m.PartSize()[Z]))
}
