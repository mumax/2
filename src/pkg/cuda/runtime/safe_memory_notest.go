// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

import (
	"testing"
	"runtime"
)


const SIZE = 1024 * 1024

func TestCopyEmptyArray(t *testing.T) {
	runtime.LockOSThread()
	size := 0
	host1 := make([]float32, size)
	dev1 := NewFloat32Array(size)
	dev2 := NewFloat32Array(size)
	host2 := make([]float32, size)

	CopyHostToDevice(dev1, host1)
	CopyDeviceToDevice(dev2, dev1)
	CopyDeviceToHost(host2, dev2)
}


func TestCopyFromTo(t *testing.T) {
	runtime.LockOSThread()
	size := SIZE
	host1 := make([]float32, size)
	for i := 0; i < len(host1)/2; i++ {
		host1[i] = float32(i)
	}
	dev1 := NewFloat32Array(size)
	dev2 := NewFloat32Array(size)
	host2 := make([]float32, size)

	CopyHostToDevice(dev1, host1)
	CopyDeviceToDevice(dev2, dev1)
	CopyDeviceToHost(host2, dev2)
	for i := 0; i < len(host1)/2; i++ {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
}

func TestCopy(t *testing.T) {
	runtime.LockOSThread()
	size := SIZE
	host1 := make([]float32, size)
	for i := 0; i < len(host1)/2; i++ {
		host1[i] = float32(i)
	}
	dev1 := NewFloat32Array(size)
	dev2 := NewFloat32Array(size)
	host2 := make([]float32, size)
	host3 := make([]float32, size)

	Copy(dev1, host1)
	Copy(dev2, dev1)
	Copy(host2, dev2)
	Copy(host3, host2)
	for i := 0; i < len(host1)/2; i++ {
		if host3[i] != float32(i) {
			t.Fail()
		}
	}
}

func TestCopyFloat32Array(t *testing.T) {
	runtime.LockOSThread()
	size := SIZE
	host1 := make([]float32, size)
	for i := 0; i < len(host1)/2; i++ {
		host1[i] = float32(i)
	}
	dev1 := NewFloat32Array(size)
	dev2 := NewFloat32Array(size)
	host2 := make([]float32, size)

	CopyFloat32ArrayToDevice(dev1, host1)
	CopyDeviceToDevice(dev2, dev1)
	CopyDeviceToFloat32Array(host2, dev2)
	for i := 0; i < len(host1)/2; i++ {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
}
