// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements memory management.

//#include <cuda_runtime.h>
import "C"
import "unsafe"

import (
	"reflect"
	"fmt"
)


// High-level safe memory copy.
// Copies src to dest, each of which can be either a *cuda.Array (on the device)
// or a Go Array or slice type (on the host).
//
// E.g.:
// host := make([]float32, 100)
// dev := NewArray(400)
// Copy(host, dev)
// Copy(dev, host)
// Copy(dev, dev)
// Copy(host, host)
//
// The host slice should contain non-pointer values like float32s, ints, ...
// It is valid to copy pointer-values like *floats to the device, but they
// have little meaning there.
//
// Source size must be equal to dest size (checked at runtime).
func Copy(dest, src interface{}) {
	destArr, destIsArr := dest.(Array)
	srcArr, srcIsArr := src.(Array)
	switch {
	case destIsArr && srcIsArr:
		CopyDeviceToDevice(destArr, srcArr)
	case !destIsArr && srcIsArr:
		CopyDeviceToHost(dest, srcArr)
	case destIsArr && !srcIsArr:
		CopyHostToDevice(destArr, src)
	case !destIsArr && !srcIsArr:
		CopyHostToHost(dest, src)
	}
}


// Specialized version of Copy() to copy to the device form
// an arbitrary array or slice type on the host.
// E.g.: CopyHostToDevice(cuda.NewArray(1000), make([]float32, 250))
// This function provides more compile-time type safety than Copy()
func CopyHostToDevice(dest Array, src interface{}) {
	srcVal := reflect.NewValue(src)
	srcArrVal, ok := srcVal.(reflect.ArrayOrSliceValue)
	if !ok {
		panic("cuda.CopyHostToDevice: source must be array or slice")
	}
	if srcArrVal.Len() == 0 {
		if dest.Bytes() == 0 {
			return
		} else {
			panic("cuda.CopyHostToDevice: source size (" + fmt.Sprint(srcArrVal.Len()) + "B)" +
				" != destination size (" + fmt.Sprint(dest.Bytes()) + "B)")
		}
	}
	srcElem0Val := srcArrVal.Elem(0)
	addrElem0 := srcElem0Val.UnsafeAddr()
	sizeofSrc := int(srcElem0Val.Type().Size())
	bytes := sizeofSrc * srcArrVal.Len()
	if bytes != dest.Bytes() {
		panic("cuda.CopyHostToDevice: source size (" + fmt.Sprint(bytes) + "B)" +
			" != destination size (" + fmt.Sprint(srcArrVal.Len()) + "B)")
	}
	Memcpy(dest.Pointer(), addrElem0, bytes, MemcpyHostToDevice)
}


// Specialized version of Copy() to copy to an arbitrary array or slice type on the host,
// from the device.
// E.g.: CopyDeviceToHost(make([]float32, 250), cuda.NewArray(1000))
// This function provides more compile-time type safety than Copy()
func CopyDeviceToHost(dest interface{}, src Array) {
	dstVal := reflect.NewValue(dest)
	dstArrVal, ok := dstVal.(reflect.ArrayOrSliceValue)
	if !ok {
		panic("cuda.CopyDeviceToHost: destination must be array or slice")
	}
	if dstArrVal.Len() == 0 {
		if src.Bytes() == 0 {
			return
		} else {
			panic("cuda.CopyDeviceToHost: source size (" + fmt.Sprint(src.Bytes()) + "B)" +
				" != destination size (" + fmt.Sprint(dstArrVal.Len()) + "B)")
		}
	}
	dstElem0Val := dstArrVal.Elem(0)
	addrElem0 := dstElem0Val.UnsafeAddr()
	sizeofDst := int(dstElem0Val.Type().Size())
	dstBytes := sizeofDst * dstArrVal.Len()
	if src.Bytes() != dstBytes {
		panic("cuda.CopyDeviceToHost: source size (" + fmt.Sprint(src.Bytes()) + "B)" +
			" != destination size (" + fmt.Sprint(dstBytes) + "B)")
	}
	Memcpy(addrElem0, src.Pointer(), src.Bytes(), MemcpyDeviceToHost)
}


// Specialized version of Copy(), device to device.
// This function provides more compile-time type safety than Copy()
func CopyDeviceToDevice(dest, src Array) {
	if src.Bytes() != dest.Bytes() {
		panic("cuda.CopyDeviceToDevice: source size (" + fmt.Sprint(src.Bytes()) + "B)" +
			" != destination size (" + fmt.Sprint(dest.Bytes()) + "B)")
	}
	Memcpy(dest.Pointer(), src.Pointer(), src.Bytes(), MemcpyDeviceToDevice)
}


// Specialized version of Copy(), host-to-host. 
// src and dest must be array or slice types.
// This function provides more compile-time type safety than Copy()
func CopyHostToHost(dest, src interface{}) {
	srcVal := reflect.NewValue(src)
	srcArrVal, ok1 := srcVal.(reflect.ArrayOrSliceValue)
	if !ok1 {
		panic("cuda.CopyHostToHost: source must be array or slice")
	}
	srcElem0Val := srcArrVal.Elem(0)
	srcAddrElem0 := srcElem0Val.UnsafeAddr()
	sizeofSrc := int(srcElem0Val.Type().Size())
	srcBytes := sizeofSrc * srcArrVal.Len()

	dstVal := reflect.NewValue(dest)
	dstArrVal, ok2 := dstVal.(reflect.ArrayOrSliceValue)
	if !ok2 {
		panic("cuda.CopyHostToHost: destination must be array or slice")
	}
	dstElem0Val := dstArrVal.Elem(0)
	dstAddrElem0 := dstElem0Val.UnsafeAddr()
	sizeofDst := int(dstElem0Val.Type().Size())
	dstBytes := sizeofDst * dstArrVal.Len()
	if dstBytes != srcBytes {
		panic("cuda.CopyDeviceToDevice: source size (" + fmt.Sprint(srcBytes) + "B)" +
			" != destination size (" + fmt.Sprint(dstBytes) + "B)")
	}
	Memcpy(dstAddrElem0, srcAddrElem0, srcBytes, MemcpyHostToHost)
}


// Specialized version of CopyHostToDevice for host []float32 arrays.
// This function provides more compile-time type safety than Copy()
func CopyFloat32ArrayToDevice(dest Array, src []float32) {
	N := len(src) * SIZEOF_FLOAT32
	if N > dest.Bytes() {
		panic("cuda.CopyFloat32ArrayToDevice: source size (" + fmt.Sprint(N) + "B)" +
			" > destination size (" + fmt.Sprint(dest.Bytes()) + "B)")
	}
	Memcpy(dest.Pointer(), uintptr(unsafe.Pointer(&src[0])), N, MemcpyHostToDevice)
}


// Specialized version of CopyDeviceToHost for host []float32 arrays.
// This function provides more compile-time type safety than Copy()
func CopyDeviceToFloat32Array(dest []float32, src Array) {
	N := src.Bytes()
	destBytes := len(dest) * SIZEOF_FLOAT32
	if N > destBytes {
		panic("cuda.CopyDeviceToFloat32Array: source size (" + fmt.Sprint(N) + "B)" +
			" > destination size (" + fmt.Sprint(destBytes) + "B)")
	}
	Memcpy(uintptr(unsafe.Pointer(&dest[0])), src.Pointer(), N, MemcpyDeviceToHost)
}
