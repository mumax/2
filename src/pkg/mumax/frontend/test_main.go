//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements mumax2's -test

import (
	. "mumax/common"
	"mumax/gpu"
	"time"
)

func testMain() {
	size := []int{10, 1024, 1024}
	a := gpu.NewArray(1, size)
	defer a.Free()
	b := gpu.NewArray(1, size)
	defer b.Free()

	println("start")
	N := 2000
	start := time.Nanoseconds()

	for i := 0; i < N; i++ {
		a.CopyFromDevice(b)
	}

	t := float64(time.Nanoseconds()-start) / 1e9
	//bw := float64(Prod(size) * SIZEOF_FLOAT * N) / t
	println("stop")
	Log("CUDA bandwidth:", t, "GB/s")
}
