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
)

// A quantity represents a scalar/vector/tensor field,value or mask.
//
// * By "value" we mean a single, space-independent scalar,vector or tensor.
// * By "field" we mean a space-dependent field of scalars, vectors or tensors.
// * By "mask" we mean the point-wise multiplication of a field by a value.
//
// Typically a mask represents A(r) * f(t), a pointwise multiplication
// of an N-vector function of space A(r) by an N-vector function of time f(t).
// A(r) is an array, f(t) is the multiplier which will be updated every time step.
// When a mask's array contains NULL pointers for each gpu, the mask is independent of space. 
// The array is then interpreted as 1(r), the unit field. In this way, masks that happen to be
// constant values in space (homogeneous) can be efficiently represented. 
// TODO: deduplicate identical mask arrays by setting identical pointers?
//
// Quantities are the nodes of an acyclic graph representing the differential
// equation to be solved.
type Quant struct {
	name       string      // Unique identifier
	array      *gpu.Array  // Underlying array, may be nil. Holds nil pointers for space-independent quantity
	multiplier []float32   // Point-wise multiplication coefficients for array, may be nil
	nComp      int         // Number of components. Defines whether it is a SCALAR, VECTOR, TENSOR,...
	upToDate   bool        // Flags if this quantity needs to be updated
	updateSelf Updater     // Called to update this quantity
	children   []*Quant    // Quantities this one depends on
	parents    []*Quant    // Quantities that depend on this one
	buffer     *host.Array // Host buffer for copying from/to the GPU array
	desc       string      // Human-readable description
	kind       QuantKind   // VALUE, FIELD or MASK
}


//____________________________________________________________________ init


// Returns a new quantity. See Quant.init().
func newQuant(name string, nComp int, size3D []int, kind QuantKind, desc ...string) *Quant {
	q := new(Quant)
	q.init(name, nComp, size3D, kind, desc...)
	return q
}


// Number of components.
const (
	SCALAR   = 1
	VECTOR   = 3
	SYMMTENS = 6
	TENS     = 9
)


// Initiates a field with nComp components and array size size3D.
// When size3D == nil, the field is space-independent (homogeneous) and the array will
// hold NULL pointers for each of the GPU parts.
// When multiply == false no multiplier will be allocated,
// indicating this quantity should not be post-multiplied.
// multiply = true
func (q *Quant) init(name string, nComp int, size3D []int, kind QuantKind, desc ...string) {
	Assert(nComp > 0)
	Assert(size3D == nil || len(size3D) == 3)

	q.name = name
	q.nComp = nComp
	q.kind = kind

	switch kind {
	case FIELD:
		q.array = gpu.NewArray(nComp, size3D)
		q.multiplier = nil
	case MASK:
		q.array = gpu.NilArray(nComp, size3D)
		q.multiplier = ones(nComp)
	case VALUE:
		q.array = nil
		q.multiplier = zeros(nComp)
	default:
		panic(Bug("Quant.init kind"))
	}

	q.updateSelf = new(NopUpdater)

	const CAP = 2
	q.children = make([]*Quant, 0, CAP)
	q.parents = make([]*Quant, 0, CAP)

	// concatenate desc strings
	buf := ""
	for i, str := range desc {
		if i > 0 {
			str += " "
		}
		buf += str
	}
	q.desc = buf
}

// array with n 1's.
func ones(n int) []float32 {
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	return ones
}


// array with n 0's.
func zeros(n int) []float32 {
	zeros := make([]float32, n)
	for i := range zeros {
		zeros[i] = 0
	}
	return zeros
}


//____________________________________________________________________ set


//
func (q *Quant) SetField(field *host.Array) {
	checkKind(q, FIELD)
	q.Array().CopyFromHost(field)
	q.Invalidate() //!
}

func (q *Quant) SetValue(val []float32) {
	checkKind(q, VALUE)
	Assert(len(val) == q.nComp)
	for i, v := range val {
		q.multiplier[i] = v
	}
	q.Invalidate() //!
}


func checkKind(q *Quant, kind QuantKind) {
	if q.kind != kind {
		panic(InputErr(q.name + " is not " + kind.String() + "but " + q.kind.String()))
	}
}

//// Sets the value to a space-independent scalar.
//// The quantity must have been first initialized as scalar.
//// If it was previously space-dependent, the array is freed.
//func (q *Quant) SetScalar(value float32) {
//	if q.array != nil {
//		q.array.Free()
//		q.array = nil
//	}
//
//	if len(q.multiplier) != 1 {
//		panic(InputErr(fmt.Sprintf(q.Name(), "has", q.NComp(), "components")))
//	}
//
//	q.multiplier[0] = value
//}


//____________________________________________________________________ get

// Gets the name
func (q *Quant) Name() string {
	return q.name
}

// Gets the number of components
func (q *Quant) NComp() int {
	return q.nComp
}


// Gets the GPU array.
func (q *Quant) Array() *gpu.Array {
	//	if q.array == nil {
	//		if q.Size3D() == nil{
	//			q.array = gpu.NilArray(q.NComp(), q.Size3D())
	//		}else{
	//			Debug("alloc ", q.Name(), q.NComp(), "x", q.Size3D())
	//			q.array = gpu.NewArray(q.NComp(), q.Size3D())
	//}
	//	}
	return q.array
}

// Gets a host array for buffering the GPU array, initializing it if necessary.
func (q *Quant) Buffer() *host.Array {
	if q.buffer == nil {
		Debug("buffer ", q.Name(), q.NComp(), "x", q.Array().Size3D())
		q.buffer = host.NewArray(q.NComp(), q.Array().Size3D())
	}
	return q.buffer
}


func (q *Quant) IsSpaceDependent() bool {
	return q.array != nil && q.array.DevicePtr()[0] != 0
}


// If the quantity represents a space-independent scalar, return its value.
//func (q *Quant) ScalarValue() float32 {
//if q.IsSpaceDependent() {
//panic(Bug("not a scalar"))
//}
//return q.multiplier[0]
//}


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
