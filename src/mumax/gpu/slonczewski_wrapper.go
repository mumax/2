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

func LLSlon(stt *Array, m *Array, msat *Array, p *Array, j *Array, alpha *Array, pMul []float64, jMul []float64, lambda2 float32, beta_prime float32, pre_field float32, worldSize []float64, alphaMul []float64) {

	// Bookkeeping
	CheckSize(p.Size3D(), m.Size3D())
	Assert(j.NComp() == 3)
    Assert(msat.NComp() == 1)
    Assert(alpha.NComp() == 1)
    
	// Calling the CUDA functions
	C.slonczewski_async(
		(**C.float)(unsafe.Pointer(&(stt.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(stt.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(msat.Comp[X].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(p.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(p.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(p.Comp[Z].Pointers()[0]))),
		
		(**C.float)(unsafe.Pointer(&(j.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(j.Comp[Z].Pointers()[0]))),

        (**C.float)(unsafe.Pointer(&(alpha.Comp[X].Pointers()[0]))),        
        
		(C.float)(pMul[X]),
		(C.float)(pMul[Y]),
		(C.float)(pMul[Z]),
		
		(C.float)(jMul[X]),
		(C.float)(jMul[Y]),
		(C.float)(jMul[Z]),
		
		(C.float)(lambda2),
		(C.float)(beta_prime),
		(C.float)(pre_field),

        (C.float)(worldSize[X]),
        (C.float)(worldSize[Y]),
        (C.float)(worldSize[Z]),
        
        (C.float)(alphaMul[X]),
        
		(C.int)(m.PartLen3D()),
		(*C.CUstream)(unsafe.Pointer(&(stt.Stream[0]))))
}
