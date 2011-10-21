// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package fft

//#include <cuda.h>
//#include <cufft.h>
import "C"

func Plan1d(nx int, typ Type, batch int) Handle {
	var handle C.cufftHandle
	err := Result(C.cufftPlan1d(&handle, C.int(nx), C.cufftType(typ), C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}
