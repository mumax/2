//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"testing"
)


// Test repeated alloc/free.
func TestArrayAlloc(t *testing.T) {
	N := BIG / 4
	size := []int{1, 1, N}
	for i := 0; i < 50; i++ {
		t := NewArray(3, size)
		t.Free()
	}
}


// Should init to zeros
func TestArrayInit(test *testing.T) {
	size := []int{4, 8, 16}
	host1 := host.NewArray(3, size)
	dev1 := NewArray(3, size)
	if dev1.Len() != 3*Prod(size) {
		test.Fail()
	}

	l1 := host1.List
	for i := range l1 {
		l1[i] = float32(i)
	}

	dev1.CopyToHost(host1)
	//host1.CopyFromDevice(dev1)

	l2 := host1.List
	for i := range l1 {
		if l2[i] != 0 {
			test.Fail()
		}
	}
}


func TestArrayCopy(test *testing.T) {
	size := []int{4, 8, 16}
	host1, host2 := host.NewArray(3, size), host.NewArray(3, size)
	dev1, dev2 := NewArray(3, size), NewArray(3, size)

	l1 := host1.List
	for i := range l1 {
		l1[i] = float32(i)
	}

	dev1.CopyFromHost(host1)
	dev2.CopyFromDevice(dev1)
	dev2.CopyToHost(host2)
	//host2.CopyFromDevice(dev2)

	l2 := host1.List
	for i := range l1 {
		if l2[i] != float32(i) {
			test.Fail()
		}
	}
}
