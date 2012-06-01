//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for t_baryakhtar.cu
// Author: Mykola Dvornik, Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)


func LLGBtAsync(t *Array, M *Array, h *Array, msat0 *Array, msat0Mul float32, lambda float32, lambda_e float32, cellsizeX float32,cellsizeY float32, cellsizeZ float32, pbc []int) {

	// Bookkeeping 
	CheckSize(h.Size3D(), M.Size3D())
	//CheckSize(msat.Size3D(), m.Size3D())
	
	//Assert(l.NComp() == 1)
	Assert(h.NComp() == 3)
	
	/*if t.PartSize()[X] < 4 || t.PartSize()[Y] < 4 || t.PartSize()[Z] < 4 {
	    panic("For LLB dimensions should have >= 4 cells!")
	}*/
	// Calling the CUDA functions
	C.tbaryakhtar_async(
		(**C.float)(unsafe.Pointer(&(t.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(t.Comp[Z].Pointers()[0]))),
    
		(**C.float)(unsafe.Pointer(&(M.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(M.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(M.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(msat0.Comp[X].Pointers()[0]))),
		(C.float)(msat0Mul),
		
		(C.float)(lambda),
		(C.float)(lambda_e),
		
		(C.int)(t.PartSize()[X]),
		(C.int)(t.PartSize()[Y]),
		(C.int)(t.PartSize()[Z]),
		
		(C.float)(cellsizeX),
		(C.float)(cellsizeY),
		(C.float)(cellsizeZ),
		
		(C.int)(pbc[X]),
		(C.int)(pbc[Y]),
		(C.int)(pbc[Z]),
		
		(*C.CUstream)(unsafe.Pointer(&(t.Stream[0]))))
}
