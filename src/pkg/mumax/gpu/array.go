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
	"sync"
	"fmt"
)

// A MuMax Array represents a 3-dimensional array of N-vectors.
//
// Layout example for a (3,4) vsplice on 2 GPUs:
// 	GPU0: X0 X1  Y0 Y1 Z0 Z1
// 	GPU1: X2 X3  Y2 Y3 Z2 Z3
type Array struct {
	pointer      []cu.DevicePtr // Pointers to array parts on each GPU.
	_size        [4]int         // INTERNAL {components, size0, size1, size2}
	size4D       []int          // {components, size0, size1, size2}
	size3D       []int          // {size0, size1, size2}
	_partSize    [3]int         // INTERNAL 
	partSize     []int          // size of the parts of the array on each gpu. 
	partLen4D    int            // total number of floats per GPU
	partLen3D    int            // total number of floats per GPU for one component
	Stream                      // multi-GPU stream for general use with this array
	Comp         []Array        // X,Y,Z components as arrays
	sync.RWMutex                // mutex for safe concurrent access to this array
}

// Initializes the array to hold a field with the number of components and given size.
// 	Init(3, 1000) // gives an array of 1000 3-vectors
// 	Init(1, 1000) // gives an array of 1000 scalars
// 	Init(6, 1000) // gives an array of 1000 6-vectors or symmetric tensors
// Storage is allocated only if alloc == true.
func (a *Array) Init(components int, size3D []int, alloc bool) {
	a.initSize(components, size3D)

	devices := getDevices()
	Ndev := len(devices)

	a.pointer = make([]cu.DevicePtr, Ndev)
	//a.devId = make([]int, Ndev)
	a.Stream = NewStream()
	if alloc {
		a.Alloc()
	}

	// initialize component arrays
	a.Comp = make([]Array, components)

	for c := range a.Comp {
		a.Comp[c].initSize(1, size3D)
		a.Comp[c].pointer = make([]cu.DevicePtr, Ndev)
		a.Comp[c].Stream = NewStream()
		a.Comp[c].Comp = nil

	}
	a.initCompPtrs()
}

func (a *Array) initCompPtrs() {

	for c := range a.Comp {
		for j := range a.Comp[c].pointer {
			//setDevice(_useDevice[j])
			start := c * a.partLen3D
			a.Comp[c].pointer[j] = cu.DevicePtr(offset(uintptr(a.pointer[j]), start*SIZEOF_FLOAT))
		}
	}
}

func (a *Array) initSize(components int, size3D []int) {
	Ndev := len(getDevices())
	Assert(components > 0)
	Assert(len(size3D) == 3)
	length3D := Prod(size3D)
	Assert(length3D > 0)
	a.partLen4D = components * length3D / Ndev
	a.partLen3D = length3D / Ndev
	Assert(size3D[Y]%Ndev == 0)

	a._size[0] = components
	for i := range size3D {
		a._size[i+1] = size3D[i]
	}
	a.size4D = a._size[:]
	a.size3D = a._size[1:]
	// Slice along the J-direction
	a._partSize[X] = a.size3D[X]
	a._partSize[Y] = a.size3D[Y] / Ndev
	a._partSize[Z] = a.size3D[Z]

}

// Returns an array which holds a field with the number of components and given size.
func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D, true)
	return t
}

// Returns an array without underlying storage. 
// This is used for space-independent quantities. These pass
// a multiplier value and a null pointer for each GPU.
// A NilArray already has null pointers for each GPU set,
// so it is more convenient than just a nil pointer of type *Array.
// See: Alloc()
func NilArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D, false)
	return t
}

// If the array has no underlying storage yet (e.g., it was
// created by NilArray()), allocate that storage.
func (a *Array) Alloc() {
	devices := getDevices()
	for i := range devices {
		setDevice(devices[i])
		Assert(a.pointer[i] == 0)
		a.pointer[i] = cu.MemAlloc(SIZEOF_FLOAT * int64(a.partLen4D))
	}
	a.Zero()
	a.initCompPtrs() // need to update the component pointers
}

// Frees the underlying storage and sets the size to zero.
func (v *Array) Free() {
	v.Stream.Destroy()
	v.Stream = nil

	for i := range v.pointer {
		setDevice(_useDevice[i])
		if v.pointer[i] != 0 {
			v.pointer[i].Free()
			v.pointer[i] = 0
		}
	}

	for i := range v._size {
		v._size[i] = 0
	}
}

// Address of part of the array on device deviceId.
func (a *Array) DevicePtr() []cu.DevicePtr {
	return a.pointer
}

// Total number of elements
func (a *Array) Len() int {
	return a._size[0] * a._size[1] * a._size[2] * a._size[3]
}

// Total number of elements per GPU
func (a *Array) PartLen4D() int {
	return a.partLen4D
}

// Number of elements per component per GPU
func (a *Array) PartLen3D() int {
	return a.partLen3D
}

// Number of components (1: scalar, 3: vector, ...).
func (a *Array) NComp() int {
	return a._size[0]
}

// Gets the i'th component as an array.
// E.g.: Component(0) is the x-component.
func (a *Array) Component(i int) *Array {
	return &(a.Comp[i])
}

// Array of pointers to parts, one per GPU.
func (a *Array) Pointers() []cu.DevicePtr {
	return a.pointer
}

// True if the array has no underlying GPU storage.
// E.g., when created by NilArray()
func (a *Array) IsNil() bool {
	return a.pointer[0] == 0
}

// Size of the vector field
func (a *Array) Size3D() []int {
	return a.size3D
}

// Copy from device array to device array.
func (dst *Array) CopyFromDevice(src *Array) {
	CheckSize(dst.size4D, src.size4D)

	d := dst.pointer
	s := src.pointer
	start := 0
	// copies run concurrently on the individual devices
	for i := range s {
		length := src.partLen4D
		cu.MemcpyDtoDAsync(cu.DevicePtr(d[i]), cu.DevicePtr(s[i]), SIZEOF_FLOAT*int64(length), dst.Stream[i])
		start += length
	}
	// Synchronize with all copies
	dst.Stream.Sync()

}

// Copy from host array to device array.
func (dst *Array) CopyFromHost(srca *host.Array) {
	CheckSize(dst.size4D, srca.Size4D)

	src := srca.Comp
	//Assert(dst.NComp() == len(src))
	// we have to work component-wise because of the data layout on the devices
	for i := range src {
		//Assert(len(dst.Comp[i]) == len(src[i])) // TODO(a): redundant
		//dst.Comp[i].CopyFromHost(src[i])

		h := src[i]
		s := dst.Comp[i].pointer
		//Assert(len(h) == len(s)) // in principle redundant
		start := 0
		for i := range s {
			length := dst.partLen3D
			//Debug("cu.MemcpyHtoD", cu.DevicePtr(s[i]), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
			cu.MemcpyHtoD(cu.DevicePtr(s[i]), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
			start += length
		}
	}
}

// Copy from device array to host array.
func (src *Array) CopyToHost(dsta *host.Array) {
	CheckSize(dsta.Size4D, src.size4D)

	dst := dsta.Comp
	//Assert(src.NComp() == len(dst))
	for i := range dst {
		//Assert(len(src.Comp[i]) == len(dst[i])) // TODO(a): redundant
		//src.Comp[i].CopyToHost(dst[i])

		h := dst[i]
		s := src.Comp[i].pointer
		//Assert(len(h) == len(s)) // in principle redundant
		start := 0
		for i := range s {
			length := src.partLen3D //s[i].length
			cu.MemcpyDtoH(cu.HostPtr(&h[start]), cu.DevicePtr(s[i]), SIZEOF_FLOAT*int64(length))
			start += length
		}

	}

}

// DEBUG: Make a freshly allocated copy on the host.
func (src *Array) LocalCopy() *host.Array {
	dst := host.NewArray(src.NComp(), src.Size3D())
	src.CopyToHost(dst)
	return dst
}

// Makes all elements zero.
func (a *Array) Zero() {
	slices := a.pointer
	for i := range slices {
		setDevice(_useDevice[i])
		cu.MemsetD32Async(slices[i], 0, int64(a.partLen4D), a.Stream[i])
	}
	a.Stream.Sync()
}

// Error message.
const MSG_ARRAY_SIZE_MISMATCH = "array size mismatch"

// Pointer arithmetic: returns ptr + bytes.
// When ptr is NULL, NULL is returned.
func offset(ptr uintptr, bytes int) uintptr {
	if ptr == 0 {
		return 0
		//panic(Bug("offsetting null pointer"))
	}
	return ptr + uintptr(bytes)
}

// Human-readable string.
func (a *Array) String() string {
	return fmt.Sprint("gpu.Array{pointers=", a.pointer, "size=", a.size4D, "}")
}
