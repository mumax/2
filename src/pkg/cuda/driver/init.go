// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver


// This file implements CUDA driver initialization

//#include <cuda.h>
import "C"

import ()

// Initialize the CUDA driver API
func Init() {
	err := Result(C.cuInit(C.uint(0))) // Flags must be 0
	if err != SUCCESS {
		panic(err)
	}
}
