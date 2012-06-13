    // Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

// Documentation was taken from the curand headers.

package curand

//#cgo LDFLAGS:-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia/ -L/usr/lib64/nvidia/ -L/usr/lib64/nvidia/lib64 -L/usr/lib/nvidia-current/ -LC:/opt/cuda/v4.2/lib/x64 -LC:/opt/cuda/v5.0/lib/x64 -lcuda -lcudart -lcurand
//#cgo CFLAGS:-I/usr/local/cuda/include/ -IC:/opt/cuda/v4.2/include -IC:/opt/cuda/v5.0/include -Wno-error
//#include <curand.h>
import "C"

import (
	"unsafe"
)

type Generator uintptr

type RngType int

const (
	PSEUDO_DEFAULT          RngType = C.CURAND_RNG_PSEUDO_DEFAULT          // Default pseudorandom generator
	PSEUDO_XORWOW           RngType = C.CURAND_RNG_PSEUDO_XORWOW           // XORWOW pseudorandom generator
	QUASI_DEFAULT           RngType = C.CURAND_RNG_QUASI_DEFAULT           // Default quasirandom generator
	QUASI_SOBOL32           RngType = C.CURAND_RNG_QUASI_SOBOL32           // Sobol32 quasirandom generator
	QUASI_SCRAMBLED_SOBOL32 RngType = C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 // Scrambled Sobol32 quasirandom generator
	QUASI_SOBOL64           RngType = C.CURAND_RNG_QUASI_SOBOL64           // Sobol64 quasirandom generator
	QUASI_SCRAMBLED_SOBOL64 RngType = C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 // Scrambled Sobol64 quasirandom generator
)

func CreateGenerator(rngType RngType) Generator {
	var rng C.curandGenerator_t
	err := Status(C.curandCreateGenerator(&rng, C.curandRngType_t(rngType)))
	if err != SUCCESS {
		panic(err)
	}
	return Generator(unsafe.Pointer(rng))
}

func (g Generator) GenerateNormal(output uintptr, n int64, mean, stddev float32) {
	err := Status(C.curandGenerateNormal(
		C.curandGenerator_t(unsafe.Pointer(g)),
		(*C.float)(unsafe.Pointer(output)),
		C.size_t(n),
		C.float(mean),
		C.float(stddev)))
	if err != SUCCESS {
		panic(err)
	}
}

func (g Generator) SetSeed(seed int64) {
	err := Status(C.curandSetPseudoRandomGeneratorSeed(C.curandGenerator_t(unsafe.Pointer(g)), _Ctype_ulonglong(seed)))
	if err != SUCCESS {
		panic(err)
	}
}
