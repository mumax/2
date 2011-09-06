//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements physical quantities represented by
// either scalar, vector or tensor fields in time and space.
// Author: Arne Vansteenkiste.

import (
	. "mumax/common"
	"mumax/gpu"
	"mumax/host"
	"fmt"
)

// Conceptually, each quantity is represented by A(r) * m(t), a pointwise multiplication
// of an N-vector function of space A(r) by an N-vector function of time m(t).
// A(r) is an array, m(t) is the multiplier.
//
// When the array is nil/NULL, the field is independent of space. The array is then
// interpreted as 1(r), the unit field. In this way, quantities that are constant
// over space (homogeneous) can be efficiently represented. These are also called values.
//
// When the array has only one component and the multiplier has N components,
// then the field has N components: a(r) * m0(t), a(r) * m1(t), ... a(r) * mN(t)
//
// Quantities are also the nodes of an acyclic graph representing the differential
// equation to be solved.
type Quant struct {
	name       string      // Unique identifier
	array      *gpu.Array  // Underlying array, nil for space-independent quantity
	multiplier []float32   // Point-wise multiplication coefficients for array
	upToDate   bool        // Flags if this quantity needs to be updated
	updateSelf Updater     // Called to update this quantity
	children   []*Quant    // Quantities this one depends on
	parents    []*Quant    // Quantities that depend on this one
	size3D     []int       // FD size (might deviate form engine size)
	buffer     *host.Array // Host buffer for copying from/to the GPU array
}


//____________________________________________________________________ init


// Returns a new quantity. See Quant.init().
func newQuant(name string, nComp int, size3D []int) *Quant {
	q := new(Quant)
	q.init(name, nComp, size3D)
	return q
}


// Initiates a field with nComp components and array size size3D.
// When size3D == nil, the field is space-independent (homogeneous).
// Storage is not yet allocated!
func (q *Quant) init(name string, nComp int, size3D []int) {
	Assert(nComp > 0)
	Assert(size3D == nil || len(size3D) == 3)

	q.name = name

	if size3D != nil {
		q.size3D = size3D
		//	q.array = gpu.NewArray(nComp, size3D)
	}

	q.multiplier = make([]float32, nComp)
	for i := range q.multiplier {
		q.multiplier[i] = 1
	}

	q.updateSelf = new(NopUpdater)

	const CAP = 2
	q.children = make([]*Quant, 0, CAP)
	q.parents = make([]*Quant, 0, CAP)
}


//____________________________________________________________________ set

// Sets the value to a space-independent scalar.
// The quantity must have been first initialized as scalar.
// If it was previously space-dependent, the array is freed.
func (q *Quant) SetScalar(value float32) {
	if q.array != nil {
		q.array.Free()
		q.array = nil
	}

	if len(q.multiplier) != 1 {
		panic(InputErr(fmt.Sprintf(q.Name(), "has", q.NComp(), "components")))
	}

	q.multiplier[0] = value
}


//____________________________________________________________________ get

// Gets the name
func (q *Quant) Name() string {
	return q.name
}

// Gets the number of components
func (q *Quant) NComp() int {
	return len(q.multiplier)
}


// Gets the 3D grid size
func (q *Quant) Size3D() []int {
	return q.size3D
}


// Gets the GPU array, initializing it if necessary
func (q *Quant) Array() *gpu.Array {
	if q.array == nil {
		Debug("alloc ", q.Name(), q.NComp(), "x", q.Size3D())
		q.array = gpu.NewArray(q.NComp(), q.Size3D())
	}
	return q.array
}

// Gets a host array for buffering the GPU array, initializing it if necessary.
func (q *Quant) Buffer() *host.Array {
	if q.buffer == nil {
		Debug("buffer ", q.Name(), q.NComp(), "x", q.Size3D())
		q.buffer = host.NewArray(q.NComp(), q.Size3D())
	}
	return q.buffer
}


// True if the quantity is a space-independent scalar
func (q *Quant) IsScalar() bool {
	return q.array == nil && len(q.multiplier) == 1
}

// True if the quantity is a space-dependent scalar field
func (q *Quant) IsScalarField() bool {
	return q.array != nil && len(q.multiplier) == 1
}

// True if the quantity is a space-independent 3-component vector
func (q *Quant) IsVector() bool {
	return q.array == nil && len(q.multiplier) == 3
}

// True if the quantity is a space-dependent 3-component vector field
func (q *Quant) IsVectorField() bool {
	return q.array != nil && len(q.multiplier) == 3
}


// If the quantity represents a space-independent scalar, return its value.
func (q *Quant) ScalarValue() float32 {
	if !q.IsScalar() {
		panic(Bug("not a scalar"))
	}
	return q.multiplier[0]
}


//____________________________________________________________________ tree walk


// If q.upToDate is false, update this node recursively.
// First Update all parents (on which this node depends),
// and then call Quant.updateSelf.Update().
// upToDate is set true.
// See: Invalidate()
func (q *Quant) Update() {
	if q.upToDate {
		return
	}

	// update parents first
	for _, p := range q.parents {
		p.Update()
	}

	// now update self
	Debug("update " + q.Name())
	q.updateSelf.Update()

	q.upToDate = true
}


// Opposite of Update. Sets upToDate flag of this node and
// all its children (which depend on this node) to false.
func (q *Quant) Invalidate() {
	q.upToDate = false
	Debug("invalidate " + q.Name())
	for _, c := range q.children {
		c.Invalidate()
	}
}


//// If the quantity represents a space-dependent field, return a host copy of its value.
//// Call FreeBuffer() to recycle it.
//func (q *Quant) FieldValue() *host.Array {
//	a := q.array
//	buffer := NewBuffer(a.NComp(), a.Size3D())
//	q.array.CopyToHost(buffer)
//	return buffer
//}

//
//func (f *Field) Free() {
//	f.array.Free()
//	f.multiplier = nil
//	f.name = ""
//}
//
//func (f *Field) Name() string {
//	return f.name
//}
//
//// Number of components of the field values.
//// 1 = scalar, 3 = vector, etc.
//func (f *Field) NComp() int {
//	return len(f.multiplier)
//}
//
//func (f *Field) IsSpaceDependent() bool {
//	return !f.array.IsNil()
//}
//
//
//func (f *Field) String() string {
//	str := f.Name() + "(" + fmt.Sprint(f.NComp()) + "-vector "
//	if f.IsSpaceDependent() {
//		str += "field"
//	} else {
//		str += "value"
//	}
//	str += ")"
//	return str
//}
//
