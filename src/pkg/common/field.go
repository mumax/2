//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements "Fields". Fields are physical quantities represented by
// either scalar, vector or tensor fields in time and space.
// Author: Arne Vansteenkiste.

import ()

// Conceptually, each field is represented by A(r) * m(t), a pointwise multiplication
// of an N-vector function of space A(r) by an N-vector function of time m(t).
// A(r) is an array, m(t) is the multiplier.
//
// When the array is nil/NULL, the field is independent of space. The array is then
// interpreted as 1(r), the unit field. In this way, quantities that are constant
// over space (homogeneous) can be represented.
//
// When the array has only one component and the multiplier has N components,
// then the field has N components: a(r) * m0(t), a(r) * m1(t), ... a(r) * mN(t)
//
type Field struct {
	array       Array // contains the size
	_multiplier [FIELD_MAX_COMP]float32
	multiplier  []float32
	name        string
}


func NewField(name string, nComp int, size3D []int) *Field{
	f := new(Field)
	f.Init(name, nComp, size3D)
	return f
}


func NewScalar(name string) *Field{
	return NewField(name, 1, nil)
}

func NewVector(name string) *Field{
	return NewField(name, 3, nil)
}

func NewScalarField(name string, size3D []int) *Field{
	return NewField(name, 1, size3D)
}

func NewVectorField(name string, size3D []int) *Field{
	return NewField(name, 3, size3D)
}

// Initiates a field with nComp components and array size size3D.
// When size3D == nil, the field is space-independent (homogeneous).
func (f *Field) Init(name string, nComp int, size3D []int) {
	Assert(nComp > 0 && nComp <= FIELD_MAX_COMP)
	Assert(size3D == nil || len(size3D) == 3)

	f.array.Free()
	if size3D != nil {
		f.array.Init(nComp, size3D)
	}

	for i := range f._multiplier {
		f._multiplier[i] = 1
	}
	f.multiplier = f._multiplier[:nComp]

	f.name = name
}


func (f *Field) Free(){
	f.array.Free()
	f.multiplier = nil
	f.name = ""
}

func (f *Field) Name()string{
	return f.name
}

// Maximum number of components of a Field.
// 1 = scalar, 3 = vector, 6 = symmetric tensor, 9 = general tensor.
const FIELD_MAX_COMP = 9
