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
	"mumax/host"
	"testing"
	"rand"
)

func TestReduce(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}

	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
	}

	a.CopyFromHost(ah)

	var cpusum float64
	for _, num := range ah.List {
		cpusum += float64(num)
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()

	gpusum := red.Sum(a)
	if gpusum != float32(cpusum) {
		test.Error("Reduce sum cpu=", cpusum, "gpu=", gpusum)
	}
}

func BenchmarkReduceSum(b *testing.B) {
	b.StopTimer()

	size := bigsize()
	a := NewArray(3, size)
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	defer a.Free()
	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		red.Sum(a)
	}
}

func BenchmarkReduceSumCPU(b *testing.B) {

	b.StopTimer()

	size := bigsize()
	a := host.NewArray(3, size)
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	b.StartTimer()
	for i := 0; i < b.N; i++ {

		var cpusum float64
		for _, num := range a.List {
			cpusum += float64(num)
		}

	}
}
