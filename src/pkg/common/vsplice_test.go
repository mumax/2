//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Author: Arne Vansteenkiste

package common

import (
	"testing"
)



// Test vsplice alloc/free
func TestVSpliceAlloc(t *testing.T) {
	N := BIG / 4
	// Test repeated alloc + free
	for i := 0; i < 20; i++ {
		s := NewVSplice(3, N)
		if s.List.Len() != 3*N {
			t.Fail()
		}
		s.Free()
	}
}
//
//
//func TestVSpliceCopy(t *testing.T) {
//	N := 1024
//	a := make([]float32, 3*N)
//	for i := range a {
//		a[i] = float32(i)
//	}
//	b := make([]float32, 3*N)
//	A := NewVSplice(3, N)
//	defer A.Free()
//	B := NewVSplice(3, N)
//	defer B.Free()
//
//	A.CopyFromHost(a)
//	B.CopyFromDevice(A)
//	B.CopyToHost(b)
//
//	for i := range b {
//		if b[i] != float32(i) {
//			t.Fail()
//		}
//	}
//}
//
//
//func BenchmarkVSpliceCopy(b *testing.B) {
//	b.StopTimer()
//	N := BIG / 8
//	b.SetBytes(3 * int64(N) * 4)
//	A := NewVSplice(3, N)
//	defer A.Free()
//	B := NewVSplice(3 * N)
//	defer B.Free()
//
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		B.CopyFromDevice(A)
//	}
//}
