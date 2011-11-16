//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// DO NOT USE TEST.FATAL: -> runtime.GoExit -> context switch -> INVALID CONTEXT!

package gpu

// Author: Arne Vansteenkiste

import (
. "mumax/common"
cu "cuda/driver"
"testing"
"fmt"
)

func BenchmarkCopyDtoD(bench *testing.B) {

	if NDevice() < 2 {
		fmt.Println("Skipping BenchmarkCopyDtoD")
	}

	bench.StopTimer()
	N1 := 1024
	N2 := 1024 
	size := []int{1, N1, N2}

	a := NewArray(1, size)
	defer a.Free()

	ptr1 := a.pointer[0]
	ptr2 := a.pointer[1]

	bytes:=SIZEOF_FLOAT*int64(a.PartLen3D())
	cu.MemcpyDtoD(ptr2, ptr1, bytes)
	bench.SetBytes(bytes)
	bench.StartTimer()
	for i:=0; i<bench.N; i++{
		cu.MemcpyDtoD(ptr2, ptr1, SIZEOF_FLOAT*int64(a.PartLen3D()))
	}
}
