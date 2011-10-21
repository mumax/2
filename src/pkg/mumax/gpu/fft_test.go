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
	"testing"
	"fmt"
)

func TestCopyPadZ(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size1 := []int{1, 4, 8}
	size2 := []int{1, 4, 8 + 2}

	a := NewArray(1, size1)
	defer a.Free()
	ah := a.LocalCopy()

	b := NewArray(1, size2)
	defer b.Free()

	for i := range ah.List {
		ah.List[i] = float32(i)
	}

	a.CopyFromHost(ah)

	CopyPadZ(b, a)

	bh := b.LocalCopy()

	fmt.Println("CopyPadZ", bh.Array)

	//	for i := range sum.List {
	//		if sum.List[i] != ah.List[i]+bh.List[i] {
	//			if !test.Failed() {
	//				test.Error(sum.List[i], "!=", ah.List[i], "+", bh.List[i])
	//			}
	//		}
	//	}
}

func BenchmarkCopyPadZ(b *testing.B) {
	b.StopTimer()

	size := bigsize()
	a := NewArray(3, size)
	a2 := NewArray(3, []int{size[0], size[1], size[2] + 2})
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	defer a.Free()
	defer a2.Free()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		CopyPadZ(a2, a)
	}
}
