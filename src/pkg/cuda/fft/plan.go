// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package fft

//#include <cufft.h>
import "C"

import (
	"unsafe"
)

// FFT plan handle, reference type to a plan
type Handle uintptr

// 1D FFT plan
func Plan1d(nx int, typ Type, batch int) Handle {
	var handle C.cufftHandle
	err := Result(C.cufftPlan1d(
		&handle, C.int(nx),
		C.cufftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}

// Execute Complex-to-Complex plan
func (plan Handle) ExecC2C(idata, odata uintptr, direction int) {
	err := Result(C.cufftExecC2C(
		C.cufftHandle(plan),
		(*C.cufftComplex)(unsafe.Pointer(idata)),
		(*C.cufftComplex)(unsafe.Pointer(odata)),
		C.int(direction)))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Real-to-Complex plan
func (plan Handle) ExecR2C(idata, odata uintptr){
	err := Result(C.cufftExecR2C(
		C.cufftHandle(plan),
		(*C.cufftReal)(unsafe.Pointer(idata)),
		(*C.cufftComplex)(unsafe.Pointer(odata))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Complex-to-Real plan
func (plan Handle) ExecC2R(idata, odata uintptr){
	err := Result(C.cufftExecC2R(
		C.cufftHandle(plan),
		(*C.cufftComplex)(unsafe.Pointer(idata)),
		(*C.cufftReal)(unsafe.Pointer(odata))))
	if err != SUCCESS {
		panic(err)
	}
}
