//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for Q.cu
// Author: Mykola Dvornik

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func Q3TM_async(Qi *Array, Ti*Array, Tj *Array, Tk *Array, Q *Array, gamma_i *Array, Gij *Array, Gik *Array, k *Array, QMul []float64, gamma_iMul []float64, GijMul []float64, GikMul []float64, kMul []float64, isCofT int, cs []float64, pbc []int) {

	// Calling the CUDA functions
	C.Q3TM_async(
	    (**C.float)(unsafe.Pointer(&(Qi.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Ti.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tj.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tk.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Q.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(gamma_i.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gij.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Gik.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(k.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(QMul[0])),
        (C.float)(float32(gamma_iMul[0])),
        (C.float)(float32(GijMul[0])),
        (C.float)(float32(GikMul[0])),
        (C.float)(float32(kMul[0])),
        
        (C.int)(isCofT),
        
        (C.int)(Qi.PartSize()[X]),
		(C.int)(Qi.PartSize()[Y]),
		(C.int)(Qi.PartSize()[Z]),
        
        (C.float)(float32(cs[X])),
        (C.float)(float32(cs[Y])),
        (C.float)(float32(cs[Z])),
        
        (C.int)(pbc[X]),
        (C.int)(pbc[Y]),
        (C.int)(pbc[Z]),
	    (*C.CUstream)(unsafe.Pointer(&(Qi.Stream[0]))),
	    
	    )
}

func Q2TM_async(Qi *Array, Ti*Array, Tj *Array, Q *Array, gamma_i *Array, Gij *Array, k *Array, QMul []float64, gamma_iMul []float64, GijMul []float64, kMul []float64, isCofT int, cs []float64, pbc []int) {

	// Calling the CUDA functions
	C.Q2TM_async(
	    (**C.float)(unsafe.Pointer(&(Qi.Comp[X].Pointers()[0]))),
	    
        (**C.float)(unsafe.Pointer(&(Ti.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(Tj.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Q.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(gamma_i.Comp[X].Pointers()[0]))),
        
        (**C.float)(unsafe.Pointer(&(Gij.Comp[X].Pointers()[0]))),
        (**C.float)(unsafe.Pointer(&(k.Comp[X].Pointers()[0]))),
        
        (C.float)(float32(QMul[0])),
        (C.float)(float32(gamma_iMul[0])),
        (C.float)(float32(GijMul[0])),
        (C.float)(float32(kMul[0])),
        
        (C.int)(isCofT),
        
        (C.int)(Qi.PartSize()[X]),
		(C.int)(Qi.PartSize()[Y]),
		(C.int)(Qi.PartSize()[Z]),
        
        (C.float)(float32(cs[X])),
        (C.float)(float32(cs[Y])),
        (C.float)(float32(cs[Z])),
        
        (C.int)(pbc[X]),
        (C.int)(pbc[Y]),
        (C.int)(pbc[Z]),
	    (*C.CUstream)(unsafe.Pointer(&(Qi.Stream[0]))),
	    
	    )
}

//func Qs_async(Qs *Array, Te *Array, Tl *Array, Ts *Array, Cs *Array, Gsl *Array, Ges *Array, CsMul []float64, GslMul []float64, GesMul []float64) {
//	// Calling the CUDA functions
//	C.Qs_async(
//	    (**C.float)(unsafe.Pointer(&(Qs.Comp[X].Pointers()[0]))),
//	    
//        (**C.float)(unsafe.Pointer(&(Te.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Tl.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Ts.Comp[X].Pointers()[0]))),
//        
//        (**C.float)(unsafe.Pointer(&(Cs.Comp[X].Pointers()[0]))),
//        
//        (**C.float)(unsafe.Pointer(&(Gsl.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Ges.Comp[X].Pointers()[0]))),
//        
//        (C.float)(float32(CsMul[0])),
//        (C.float)(float32(GslMul[0])),
//        (C.float)(float32(GesMul[0])),
//        
//	    (*C.CUstream)(unsafe.Pointer(&(Qs.Stream[0]))),
//	    (C.int)(Qs.partLen3D))
//}

//func Ql_async(Ql *Array, Te *Array, Tl *Array, Ts *Array, Cl *Array, Gel *Array, Gsl *Array, ClMul []float64, GelMul []float64, GslMul []float64) {
//	// Calling the CUDA functions
//	C.Ql_async(
//	    
//	    (**C.float)(unsafe.Pointer(&(Ql.Comp[X].Pointers()[0]))),
//	    
//        (**C.float)(unsafe.Pointer(&(Te.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Tl.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Ts.Comp[X].Pointers()[0]))),
//        
//        (**C.float)(unsafe.Pointer(&(Cl.Comp[X].Pointers()[0]))),
//        
//        (**C.float)(unsafe.Pointer(&(Gel.Comp[X].Pointers()[0]))),
//        (**C.float)(unsafe.Pointer(&(Gsl.Comp[X].Pointers()[0]))),
//        
//        (C.float)(float32(ClMul[0])),
//        (C.float)(float32(GelMul[0])),
//        (C.float)(float32(GslMul[0])),
//        
//	    (*C.CUstream)(unsafe.Pointer(&(Ql.Stream[0]))),
//	    (C.int)(Ql.partLen3D))
//}
