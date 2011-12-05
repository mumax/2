// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package curand

//#include <curand.h>
import "C"

import (
	"unsafe"
)



type Generator uintptr 

type RngType int


const(

)

func CreateGenerator(rngType RngType) Generator{
	var rng C.curandGenerator_t
	err := Status(C.curandCreateGenerator(&rng, C.curandRngType_t(rngType)))
	if err != SUCCESS {
		panic(err)
	}
	return Generator(unsafe.Pointer(rng))
}

