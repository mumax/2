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
	"sync"
	"fmt"
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
	name        string            // Unique identifier
	array       *gpu.Array        // Underlying array of dimensionless values typically of order 1. Holds nil pointers for space-independent quantities.
	multiplier  []float64         // Point-wise multiplication coefficients for array, dimensionfull.
	nComp       int               // Number of components. Defines whether it is a SCALAR, VECTOR, TENSOR,...
	upToDate    bool              // Flags if this quantity needs to be updated
	updater     Updater           // Called to update this quantity
	children    map[string]*Quant // Quantities this one depends on, indexed by name
	parents     map[string]*Quant // Quantities that depend on this one, indexed by name
	desc        string            // Human-readable description
	unit        Unit              // Unit of the multiplier value, e.g. A/m.
	kind        QuantKind         // VALUE, FIELD or MASK
	updates     int               // Number of times the quantity has been updated (for debuggin)
	invalidates int               // Number of times the quantity has been invalidated (for debuggin)
	buffer      *host.Array       // Host buffer for copying from/to the GPU array
	bufUpToDate bool              // Flags if the buffer (in RAM) needs to be updated
	bufXfers    int               // Number of times it has been copied from GPU
	bufMutex    sync.RWMutex
	Timer       // Debug/benchmarking
}

//____________________________________________________________________ init

// Returns a new quantity. See Quant.init().
func newQuant(name string, nComp int, size3D []int, kind QuantKind, unit Unit, desc ...string) *Quant {
	q := new(Quant)
	q.init(name, nComp, size3D, kind, unit, desc...)
	return q
}

// Number of components.
const (
	SCALAR   = 1 // Number
	VECTOR   = 3 // Vector
	SYMMTENS = 6 // Symmetric tensor
	TENS     = 9 // General tensor
)

// Initiates a field with nComp components and array size size3D.
// When size3D == nil, the field is space-independent (homogeneous) and the array will
// hold NULL pointers for each of the GPU parts.
// When multiply == false no multiplier will be allocated,
// indicating this quantity should not be post-multiplied.
// multiply = true
func (q *Quant) init(name string, nComp int, size3D []int, kind QuantKind, unit Unit, desc ...string) {
	Assert(nComp > 0)
	Assert(size3D == nil || len(size3D) == 3)

	q.name = name
	q.nComp = nComp
	q.kind = kind

	switch kind {
	// A FIELD is calculated by mumax itself, not settable by the user.
	// So it should not have a multiplier, but always have allocated storage.
	case FIELD:
		q.array = gpu.NewArray(nComp, size3D)
		q.multiplier = ones(nComp)
	// A MASK should always have a value (stored in the multiplier).
	// We initialize it to zero. The space-dependent mask is optinal
	// and not yet allocated.
	case MASK:
		q.array = gpu.NilArray(nComp, size3D)
		q.multiplier = zeros(nComp)
	// A VALUE is space-independent and thus does not have allocated storage.
	// The value is stored in the multiplier and initialized to zero.
	case VALUE:
		q.array = nil
		q.multiplier = zeros(nComp)
	default:
		panic(Bug("Quant.init kind"))
	}

	q.updater = new(NopUpdater)

	const CAP = 2
	q.children = make(map[string]*Quant)
	q.parents = make(map[string]*Quant)

	q.unit = unit

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
func ones(n int) []float64 {
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1
	}
	return ones
}

// array with n 0's.
func zeros(n int) []float64 {
	zeros := make([]float64, n)
	for i := range zeros {
		zeros[i] = 0
	}
	return zeros
}

//____________________________________________________________________ set

// Set the multiplier of a MASK or the value of a VALUE
func (q *Quant) SetValue(val []float64) {
	Debug("SetValue", q.name, val)
	checkKinds(q, MASK, VALUE)
	checkComp(q, len(val))
	for i, v := range val {
		q.multiplier[i] = v
	}
	q.Invalidate() //!
}

// Convenience method for SetValue([]float64{val})
func (q *Quant) SetScalar(val float64) {
	checkKind(q, VALUE)
	q.multiplier[0] = val
	q.Invalidate() //!
}

// Sets a space-dependent field.
func (q *Quant) SetField(field *host.Array) {
	checkKind(q, FIELD)
	q.Array().CopyFromHost(field)
	q.Invalidate() //!
}

// Sets the space-dependent mask array.
// Allocates GPU storage when needed.
func (q *Quant) SetMask(field *host.Array) {
	checkKind(q, MASK)
	q.assureAlloc()
	Debug(q.Name(), q.Array())
	q.Array().CopyFromHost(field)
	q.Invalidate() //!
}

//____________________________________________________________________ get

// Assuming the quantity represent a scalar value, return it as a number.
func (q *Quant) Scalar() float64 {
	q.Update()
	if q.IsSpaceDependent() {
		panic(InputErr(q.Name() + " is space-dependent, can not return it as a scalar value"))
	}
	return q.multiplier[0]
}

// Gets the name
func (q *Quant) Name() string {
	return q.name
}

// Gets the name + [unit]
func (q *Quant) FullName() string {
	unit := string(q.unit)
	if unit == "" {
		return q.name
	}
	return q.name + " [" + unit + "]"
}

// Gets the number of components
func (q *Quant) NComp() int {
	return q.nComp
}

func (q *Quant) Unit() Unit {
	return q.unit
}

// Gets the GPU array.
func (q *Quant) Array() *gpu.Array {
	return q.array
}

func (q *Quant) IsSpaceDependent() bool {
	return q.array != nil && q.array.DevicePtr()[0] != 0
}

// Transfers the quantity from GPU to host. The quantities host buffer
// is allocated when needed. The transfer is only done when needed, i.e.,
// when bufferUpToDate == false. Multiplies by the multiplier and handles masks correctly.
// Does not Update().
func (q *Quant) Buffer() *host.Array {
	if q.bufUpToDate {
		return q.buffer
	}

	//q.bufMutex.Lock()

	// allocate if needed
	array := q.Array()
	if q.buffer == nil {
		//Debug("buffer", q.Name(), q.NComp(), "x", q.Array().Size3D())
		q.buffer = host.NewArray(q.NComp(), q.Array().Size3D())
	}

	// copy
	buffer := q.buffer
	if array.IsNil() {
		for c := range buffer.Comp {
			comp := buffer.Comp[c]
			for i := range comp {
				comp[i] = float32(q.multiplier[c])
			}
		}
	} else {
		q.array.CopyToHost(q.buffer)
		q.bufXfers++
		for c := range buffer.Comp {
			if q.multiplier[c] != 1 {
				comp := buffer.Comp[c]
				for i := range comp {
					comp[i] *= float32(q.multiplier[c]) // multiply by multiplier if not 1
				}
			}
		}
	}
	q.bufUpToDate = true
	//q.bufMutex.Unlock()
	return q.buffer
}

//____________________________________________________________________ tree walk

// If q.upToDate is false, update this node recursively.
// First Update all parents (on which this node depends),
// and then call Quant.updateSelf.Update().
// upToDate is set true.
// See: Invalidate()
func (q *Quant) Update() {
	//Log("update", q.Name(), valid(!q.upToDate))
	//if q.upToDate {
	//	return
	//}

	// update parents first
	for _, p := range q.parents {
		p.Update()
	}

	// now update self
	//Log("actually update " + q.Name())
	if !q.upToDate {
		q.StartTimer()
		q.updater.Update()
		q.StopTimer()
		q.updates++
	}

	q.upToDate = true
	// Do not update buffer!
}

// Opposite of Update. Sets upToDate flag of this node and
// all its children (which depend on this node) to false.
func (q *Quant) Invalidate() {
	//Log("invalidate", q.Name(), valid(q.upToDate))
	//if !q.upToDate {
	//	return
	//}

	if q.upToDate {
		q.invalidates++
	}
	q.upToDate = false
	q.bufUpToDate = false
	Debug("invalidate " + q.Name())
	//Log("actually invalidate " + q.Name())
	for _, c := range q.children {
		c.Invalidate()
	}
}

//___________________________________________________________ 

// INTERNAL: in case of a MASK, make sure the underlying array is allocted.
// Used, e.g., when a space-independent mask gets replaced by a space-dependent one.
func (q *Quant) assureAlloc() {
	pointers := q.Array().Pointers()
	if pointers[0] == 0 {
		Debug("assureAlloc: " + q.Name())
		q.Array().Alloc()
		Debug(q.Name(), q.Array())
	}
}

// Checks if the quantity has the specified kind
// Panics if check fails.
func checkKind(q *Quant, kind QuantKind) {
	if q.kind != kind {
		panic(InputErr(q.name + " is not " + kind.String() + " but " + q.kind.String()))
	}
}

// Checks if the quantity has one of the specified kinds.
// Panics if check fails.
func checkKinds(q *Quant, kind1, kind2 QuantKind) {
	if q.kind != kind1 && q.kind != kind2 {
		panic(InputErr(q.name + " is not " + kind1.String() + " or " + kind2.String() + " but " + q.kind.String()))
	}
}

// Checks if the quantity has ncomp components.
// Panics if check fails.
func checkComp(q *Quant, ncomp int) {
	if ncomp != q.nComp {
		panic(InputErr(fmt.Sprint(q.Name(), " has ", q.nComp, " components, but ", ncomp, " are provided.")))
	}
}

func (q *Quant) String() string {
	return fmt.Sprint(q.Name(), q.Buffer().Array)
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
