//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Author: Arne Vansteenkiste

import (
	"testing"
)


// Test vsplice alloc/free
func TestVSpliceAlloc(t *testing.T) {
	N := BIG / 4
	// Test repeated alloc + free
	for i := 0; i < 20; i++ {
		s := newVSplice(3, N)
		if s.list.Len() != 3*N {
			t.Fail()
		}
		s.Free()
		if !s.IsNil(){t.Fail()}
	}
}


func TestVSpliceComponent(t *testing.T) {
	N := 100 //BIG/4
	a := make([][]float32, 3)
	b := make([][]float32, 3)
	for i := range a {
		a[i] = make([]float32, N)
		b[i] = make([]float32, N)
		for j := range a[i] {
			a[i][j] = float32(i + 1)
		}
	}

	A := newVSplice(3, N)
	defer A.Free()

	A.CopyFromHost(a)

	for i := range b {
		A.Comp[i].CopyToHost(b[i])
		for j := range b[i] {
			if b[i][j] != float32(i+1) {
				t.Error("Expected ", i+1, "got", b[i][j])
			}
		}
	}

}


func TestVSpliceCopy(t *testing.T) {
	N := BIG / 8
	a := make([][]float32, 3)
	b := make([][]float32, 3)
	for i := range a {
		a[i] = make([]float32, N)
		b[i] = make([]float32, N)
		for j := range a[i] {
			a[i][j] = float32(i+1) + 0.001*float32(j)
		}
	}

	A := newVSplice(3, N)
	defer A.Free()

	for i := range a {
		A.Comp[i].CopyFromHost(a[i])
	}

	//A.Println()

	B := newVSplice(3, N)
	defer B.Free()

	for i := range a {
		B.Comp[i].CopyFromDevice(A.Comp[i])
	}
	//B.Println()
	B.CopyFromDevice(A)
	//B.Println()

	B.CopyToHost(b)
	//fmt.Println(b)

	for i := range b {
		for j := range b[i] {
			if b[i][j] != a[i][j] {
				t.Fail() //("Expected ", a[i][j], "got", b[i][j])
			}
		}
	}
}


func BenchmarkVSpliceCopy(b *testing.B) {
	b.StopTimer()
	N := BIG / 8
	b.SetBytes(3 * int64(N) * 4)
	A := newVSplice(3, N)
	defer A.Free()
	B := newVSplice(3, N)
	defer B.Free()

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		B.CopyFromDevice(A)
	}
}
