//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrapper for Qinter.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func Qinter_async(Qi *Array, Ti *Array, Tj *Array, Gij *Array, GijMul []float64, stream Stream) {

	// Calling the CUDA functions
	C.QinterAsync(
	    (**C.float)(unsafe.Pointer(&(Qi.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Ti.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tj.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gij.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(GijMul[0])),
        (C.int)(Qi.partLen3D),
	    (*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}
