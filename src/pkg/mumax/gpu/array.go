//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements 3-dimensional arrays of N-vectors distributed over multiple GPUs.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	cu "cuda/driver"
)


// A MuMax Array represents a 3-dimensional array of N-vectors,
// transparently distributed over multiple GPUs
//
// Data layout example for a 3-component array on 2 GPUs:
// 	GPU0: X0 X1  Y0 Y1 Z0 Z1
// 	GPU1: X2 X3  Y2 Y3 Z2 Z3
//
// TODO: get device part as array?: requires gpuid[] field
type Array struct {
	devPtr    []cu.DevicePtr // Access to the portions on the different GPUs
	devStream []cu.Stream    // devStr[i]: cached stream on device i, no need to create/destroy all the time
	//devId        []int          // 
	_size        [4]int  // INTERNAL {components, size0, size1, size2}
	size4D       []int   // {components, size0, size1, size2}
	size3D       []int   // {size0, size1, size2}
	partSize     []int   // Size3D of the parts stored on each GPU, cut along the Y-axis
	_partSize    [3]int  // INTENRAL
	length4D     int     // total Number of floats
	partLength4D int     // total number of floats on each GPU
	partLength3D int     // total number of floats per component on each GPU
	Comp         []Array // x,y,z components, nil for scalar field
}


// Initializes the array to hold a field with the number of components and given size.
// 	Init(3, 1000) // gives an array of 1000 3-vectors
// 	Init(1, 1000) // gives an array of 1000 scalars
// 	Init(6, 1000) // gives an array of 1000 6-vectors or symmetric tensors
func (t *Array) InitArray(components int, size3D []int) {
	Ndev := NDevice()
	length3D := Prod(size3D)

	Assert(size3D[1]%Ndev == 0)
	Assert(length3D%Ndev == 0)

	t.initSizes(components, size3D)

	// init storage and streams
	t.devPtr = make([]cu.DevicePtr, Ndev)
	t.devStream = make([]cu.Stream, Ndev)

	slicelen := components * (length3D / Ndev)
	for i := 0; i < Ndev; i++ {
		assureContextId(i) // Switch device context if necessary
		t.devPtr[i] = cu.MemAlloc(SIZEOF_FLOAT * int64(slicelen))
		t.devStream[i] = cu.StreamCreate()
	}

	//compSliceLen := length / Ndev

	// initialize component arrays
	t.Comp = make([]Array, components)
	for c := range t.Comp {
		t.Comp[c].initSizes(1, t.size3D)
		t.Comp[c].devPtr = make([]cu.DevicePtr, Ndev) // Todo: could be block-allocated and sliced
		t.Comp[c].devStream = make([]cu.Stream, Ndev)
		for i := 0; i < Ndev; i++ {
			t.Comp[c].devPtr[i] = offset(t.devPtr[i], c*t.Comp[c].Len()*SIZEOF_FLOAT)
			t.Comp[c].devStream[i] = cu.StreamCreate()
		}
	}

	t.Zero() // initialize with zeros
}

// initializes size3D, size4D, length, ...
func (a *Array) initSizes(components int, size3D []int) {
	Assert(components > 0)
	Assert(len(size3D) == 3)

	a._size[0] = components
	for i := range size3D {
		a._size[i+1] = size3D[i]
	}
	a.size4D = a._size[:]
	a.size3D = a._size[1:]
	a._partSize[0] = a.size3D[0]
	a._partSize[1] = a.size3D[1] / NDevice() // Slice along the J-direction
	a._partSize[2] = a.size3D[2]
	a.partSize = a._partSize[:]
	a.length4D = Prod(a.size4D)
	a.partLength4D = a.length4D / NDevice()
	a.partLength3D = a.partLength4D / components
}

// Returns an array which holds a field with the number of components and given size.
func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.InitArray(components, size3D)
	return t
}


// Frees the underlying storage and invalidates all 
// array fields to avoid accidental use after Free.
func (a *Array) Free() {

	for i := range a.devPtr {
		assureContextId(i)
		cu.MemFree(&(a.devPtr[i]))
		a.devStream[i].Destroy()
	}

	a.invalidate()
}


// INTENRAL: invalidate all fields to protect against accidental use after Free()
func (a *Array) invalidate() {
	for i := range a._size {
		a._size[i] = 0 // also sets size3D, size4D to zero
	}
	for i := range a._partSize {
		a._partSize[i] = 0
	}
	for i := range a.devPtr {
		a.devPtr[i] = cu.DevicePtr(0)
		a.devStream[i] = cu.Stream(0)
	}
	a.length4D = 0
	a.partLength4D = 0
	a.size3D = nil
	a.size4D = nil
	a.partSize = nil
	a.devPtr = nil
	if a.Comp != nil {
		for i := range a.Comp {
			a.Comp[i].invalidate()
		}
	}
}


// Address of part of the array on device deviceId.
func (a *Array) DevicePtr(deviceId int) cu.DevicePtr {
	return a.devPtr[deviceId]
}

// True if unallocated/freed.
//func (a *Array) IsNil() bool {
//	return a.splice.IsNil()
//}

// Total number of elements
func (a *Array) Len() int {
	return a._size[0] * a._size[1] * a._size[2] * a._size[3]
}

// Number of components (1: scalar, 3: vector, ...).
func (a *Array) NComp() int {
	return a._size[0]
}

// Size of the vector field
func (a *Array) Size3D() []int {
	return a.size3D
}
func (dst *Array) CopyFromDevice(src *Array) {
	// test for equal size
	for i, d := range dst._size {
		if d != src._size[i] {
			panic(MSG_ARRAY_SIZE_MISMATCH)
		}
	}
	for i := range dst.devPtr {
		println("cu.MemcpyDtoDAsync", src.devPtr[i], dst.devPtr[i], SIZEOF_FLOAT*int64(dst.length4D), dst.devStream[i])
		cu.MemcpyDtoDAsync(src.devPtr[i], dst.devPtr[i], SIZEOF_FLOAT*int64(dst.length4D), dst.devStream[i])
	}
	// Synchronize with all copies
	for _, s := range dst.devStream {
		s.Synchronize()
	}

}


// Copy from host array to device array.
func (dsta *Array) CopyFromHost(srca *host.Array) {
	// test for equal size
	for i, d := range dsta._size {
		if d != srca.Size[i] {
			panic(MSG_ARRAY_SIZE_MISMATCH)
		}
	}

	// we have to work component-wise because of the data layout on the devices
	for c := range srca.Comp {
		dst := dsta.Comp[c]
		src := srca.Comp[c]

		for i := range dst.devPtr {
			println("cu.MemcpyHtoD", dst.devPtr[i], cu.HostPtr(&(src[i*dst.partLength3D])), SIZEOF_FLOAT*int64(dst.partLength3D))
			cu.MemcpyHtoD(dst.devPtr[i], cu.HostPtr(&(src[i*dst.partLength3D])), SIZEOF_FLOAT*int64(dst.partLength3D))
		}
	}

	//	src := srca.Comp
	//
	//	Assert(dst.NComp() == len(src))
	//	// we have to work component-wise because of the data layout on the devices
	//	for i := range src {
	//		//Assert(len(dst.Comp[i]) == len(src[i])) // TODO(a): redundant
	//		//dst.Comp[i].CopyFromHost(src[i])
	//
	//		h := src[i]
	//		s := dst.comp[i]
	//		//Assert(len(h) == len(s)) // in principle redundant
	//		start := 0
	//		for i := range s {
	//			length := s[i].length
	//			cu.MemcpyHtoD(cu.DevicePtr(s[i].array), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
	//			start += length
	//		}
	//	}
}


// Copy from device array to host array.
func (srca *Array) CopyToHost(dsta *host.Array) {
	// test for equal size
	for i, d := range srca._size {
		if d != dsta.Size[i] {
			panic(MSG_ARRAY_SIZE_MISMATCH)
		}
	}

	// we have to work component-wise because of the data layout on the devices
	for c := range srca.Comp {
		dst := dsta.Comp[c]
		src := srca.Comp[c]

		for i := range src.devPtr {
			cu.MemcpyDtoH(cu.HostPtr(&dst[i*src.partLength3D]), src.devPtr[i], SIZEOF_FLOAT*int64(src.partLength3D))
		}
	}

	//	dst := dsta.Comp
	//
	//	Assert(src.NComp() == len(dst))
	//	for i := range dst {
	//		//Assert(len(src.Comp[i]) == len(dst[i])) // TODO(a): redundant
	//		//src.Comp[i].CopyToHost(dst[i])
	//
	//		h := dst[i]
	//		s := src.comp[i]
	//		//Assert(len(h) == len(s)) // in principle redundant
	//		start := 0
	//		for i := range s {
	//			length := s[i].length
	//			cu.MemcpyDtoH(cu.HostPtr(&h[start]), cu.DevicePtr(s[i].array), SIZEOF_FLOAT*int64(length))
	//			start += length
	//		}
	//
	//	}

}


// DEBUG: Make a freshly allocated copy on the host.
func (src *Array) LocalCopy() *host.Array {
	dst := host.NewArray(src.NComp(), src.Size3D())
	src.CopyToHost(dst)
	return dst
}


func (a *Array) Zero() {
	// Start memsets in parallel on each device
	for i := range a.devPtr {
		assureContextId(i) // !!
		cu.MemsetD32Async(a.devPtr[i], 0, int64(a.partLength4D), a.devStream[i])
	}
	// Wait for each device to finish
	for _, s := range a.devStream {
		s.Synchronize()
	}
}


// Error message.
const MSG_ARRAY_SIZE_MISMATCH = "array size mismatch"


// INTERNAL: pointer arithmetic.
// note: Do not forget bytes = floats * SIZEOF_FLOAT.
func offset(ptr cu.DevicePtr, bytes int) cu.DevicePtr {
	return cu.DevicePtr(uintptr(ptr) + uintptr(bytes))
}
