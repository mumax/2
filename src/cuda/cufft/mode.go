// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package cufft

//#cgo LDFLAGS:-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia/ -L/usr/lib64/nvidia/ -L/usr/lib64/nvidia/lib64 -L/usr/lib/nvidia-current/ -LC:/opt/cuda/v5.0/lib/x64 -LC:/opt/cuda/v4.2/lib/x64 -lcuda -lcudart -lcufft
//#cgo CFLAGS:-I/usr/local/cuda/include/ -IC:/opt/cuda/v4.2/include -IC:/opt/cuda/v5.0/include -Wno-error
//#include <cufft.h>
import "C"

import (
	"fmt"
)

// CUFFT compatibility mode
type CompatibilityMode int

const (
	COMPATIBILITY_NATIVE          CompatibilityMode = C.CUFFT_COMPATIBILITY_NATIVE
	COMPATIBILITY_FFTW_PADDING    CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_PADDING
	COMPATIBILITY_FFTW_ASYMMETRIC CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC
	COMPATIBILITY_FFTW_ALL        CompatibilityMode = C.CUFFT_COMPATIBILITY_FFTW_ALL
)

func (t CompatibilityMode) String() string {
	if str, ok := compatibilityModeString[t]; ok {
		return str
	}
	return fmt.Sprint("CUFFT Compatibility mode with unknown number:", int(t))
}

var compatibilityModeString map[CompatibilityMode]string = map[CompatibilityMode]string{
	COMPATIBILITY_NATIVE:          "CUFFT_COMPATIBILITY_NATIVE",
	COMPATIBILITY_FFTW_PADDING:    "CUFFT_COMPATIBILITY_FFTW_PADDING",
	COMPATIBILITY_FFTW_ASYMMETRIC: "CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC",
	COMPATIBILITY_FFTW_ALL:        "CUFFT_COMPATIBILITY_FFTW_ALL"}
