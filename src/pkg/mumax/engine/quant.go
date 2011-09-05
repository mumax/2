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
type Quant struct {
	name       string
	array      *gpu.Array // Underlying array, nil for space-independent quantity
	multiplier []float32  // Point-wise multiplication coefficients for array
	updateSelf Updater    // Called to update this quantity
	upToDate   bool       // Flags if this quantity needs to be updated
	children   []*Quant   // Quantities this one depends on
	parents    []*Quant   // Quantities that depend on this one
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
func (q *Quant) init(name string, nComp int, size3D []int) {
	Assert(nComp > 0)
	Assert(size3D == nil || len(size3D) == 3)

	q.name = name

	if size3D != nil {
		q.array = gpu.NewArray(nComp, size3D)
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
