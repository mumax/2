//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for Q.cu
// Author: Mykola Dvornik, Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func Qe_async(Qe *Array, Te *Array, Tl *Array, Ts *Array, Q *Araray, gamma_e *Array, Gel *Array, Ges *Array, QMul []float64, gamma_eMul []float64, GelMul []float64, GesMul []float64) {

	// Calling the CUDA functions
	C.Qe_async(
	    (**C.float)(unsafe.Pointer(&(Qe.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Te.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tl.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Ts.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Q.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(gamma_e.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gel.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Ges.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(QMul[0])),
        (C.float)(float32(gamma_eMul[0])),
        (C.float)(float32(GelMul[0])),
        (C.float)(float32(GesMul[0])),
        
	    (*C.CUstream)(unsafe.Pointer(&(Qe.Stream[0]))))
}

func Qs_async(Qs *Array, Ts *Array, Te *Array, Tl *Array, Cs *Array, Gsl *Array, Ges *Array, CsMul []float64, GslMul []float64, GesMul []float64) {
	// Calling the CUDA functions
	C.Qs_async(
	    (**C.float)(unsafe.Pointer(&(Qs.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Te.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tl.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Ts.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Cs.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gsl.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Ges.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(CsMul[0])),
        (C.float)(float32(GslMul[0])),
        (C.float)(float32(GesMul[0])),
        
	    (*C.CUstream)(unsafe.Pointer(&(Qs.Stream[0]))))
}

func Ql_async(Ql *Array, Tl *Array, Te *Array, Ts *Array, Cl *Array, Gel *Array, Gsl *Array, ClMul []float64, GelMul []float64, GslMul []float64) {
	// Calling the CUDA functions
	C.Ql_async(
	    
	    (**C.float)(unsafe.Pointer(&(Ql.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Te.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tl.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Ts.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Cl.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gel.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Gsl.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(ClMul[0])),
        (C.float)(float32(GelMul[0])),
        (C.float)(float32(GslMul[0])),
        
	    (*C.CUstream)(unsafe.Pointer(&(Ql.Stream[0]))))
}
