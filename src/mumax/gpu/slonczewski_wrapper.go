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

//  void slonczewski_async(float** sttx, float** stty, float** sttz, 
//			 float** mx, float** my, float** mz, 
//			 float** px, float** py, float** pz,
//			 float pxMul, float pyMul, float pzMul,
//			 float aj, float bj, float Pol,
//			 float** jx, float IeMul,
//			 int NPart, 
//			 CUstream* stream)

func LLSlon(stt, m, p *Array, pMul []float64, aj, bj, Pol float32, jx *Array, IeMul float32) {

	// Bookkeeping
	//CheckSize(j.Size3D(), m.Size3D())
	CheckSize(p.Size3D(), m.Size3D())
	//CheckSize(alpha.Size3D(), m.Size3D())
	//CheckSize(Msat.Size3D(), m.Size3D())

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

		(C.float)(pMul[X]),
		(C.float)(pMul[Y]),
		(C.float)(pMul[Z]),
		(C.float)(aj),
		(C.float)(bj),
		(C.float)(Pol),

		// The program X component is the user Z component!
		(**C.float)(unsafe.Pointer(&(j.Comp[X].Pointers()[0]))),
		(C.float)(IeMul),

		(C.int)(m.PartLen3D()),
		(*C.CUstream)(unsafe.Pointer(&(stt.Stream[0]))))
}
