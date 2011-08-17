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
	"testing"
	"rand"
	"fmt"
)

func TestAddClosure(test *testing.T) {
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

	b := NewArray(3, size)
	defer b.Free()
	bh := b.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
		bh.List[i] = rand.Float32()
	}

	a.CopyFromHost(ah)
	b.CopyFromHost(bh)

	ClAdd(a, a, b)

	sum := a.LocalCopy()
	for i := range sum.List {
		if sum.List[i] != ah.List[i]+bh.List[i] {
			if !test.Failed() {
				test.Error(sum.List[i], "!=", ah.List[i], "+", bh.List[i])
			}
		}
	}
}

func TestAddCgo(test *testing.T) {
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

	b := NewArray(3, size)
	defer b.Free()
	bh := b.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
		bh.List[i] = rand.Float32()
	}

	a.CopyFromHost(ah)
	b.CopyFromHost(bh)

	Add(a, a, b)

	sum := a.LocalCopy()
	for i := range sum.List {
		if sum.List[i] != ah.List[i]+bh.List[i] {
			if !test.Failed() {
				test.Error(sum.List[i], "!=", ah.List[i], "+", bh.List[i])
			}
		}
	}
}


func BenchmarkAddClosure(bench *testing.B) {
	bench.StopTimer()
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			fmt.Println(err)
		}
	}()
	size := []int{1, 1024, 1024}

	a := NewArray(3, size)
	defer a.Free()
	b := NewArray(3, size)
	defer b.Free()

	// warm up
	ClAdd(a, a, b)

	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		ClAdd(a, a, b)
	}
}

func BenchmarkAddCgo(bench *testing.B) {
	bench.StopTimer()
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			fmt.Println(err)
		}
	}()
	size := []int{1, 1024, 1024}

	a := NewArray(3, size)
	defer a.Free()
	b := NewArray(3, size)
	defer b.Free()

	// warm up
	Add(a, a, b)

	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		Add(a, a, b)
	}
}
