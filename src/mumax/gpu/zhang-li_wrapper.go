//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for slonczewski_torque.cu
// Author: Mykola Dvornik, Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

//__declspec(dllexport)  void zhangli_async(float** sttx, float** stty, float** sttz, 
//			 float** mx, float** my, float** mz, 
//			 float** jx, float** jy, float** jz,
//			 const float pred, const float pret,
//			 const int sy, const int sz,
//			 const float csx, const float csy, const float csz,
//			 int NPart,
//			 CUstream* stream);

func LLZhangLi(stt *Array, m *Array, j *Array, msat *Array, jMul []float64, pred float32, pret float32, cellsizeX float32, cellsizeY float32, cellsizeZ float32, pbc []int) {

	// Bookkeeping
	CheckSize(j.Size3D(), m.Size3D())
	CheckSize(msat.Size3D(), m.Size3D())

	Assert(j.NComp() == 3)
	Assert(msat.NComp() == 1)
	//Debug("Part size:",m.PartSize()[X],"x",m.PartSize()[Y],"x",m.PartSize()[Z])

	// Calling the CUDA functions
	C.zhangli_async(
		(**C.float)(unsafe.Pointer(&(stt.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(j.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(msat.Comp[X].Pointers()[0]))),

		(C.float)(jMul[X]),
		(C.float)(jMul[Y]),
		(C.float)(jMul[Z]),

		(C.float)(pred),
		(C.float)(pret),

		(C.int)(m.PartSize()[X]),
		(C.int)(m.PartSize()[Y]),
		(C.int)(m.PartSize()[Z]),

		(C.float)(cellsizeX),
		(C.float)(cellsizeY),
		(C.float)(cellsizeZ),

		(C.int)(pbc[X]),
		(C.int)(pbc[Y]),
		(C.int)(pbc[Z]),

		(*C.CUstream)(unsafe.Pointer(&(stt.Stream[0]))))

	stt.Stream.Sync()
}
