//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	"mumax/gpu"
)


// Node serves as a superclass for all Quants, 
// implementing the common functionality.
// When a type embeds Node, it only needs to implement
// Update() to fulfill the Quant interface.
type Node struct {
	children []Quant
	parents  []Quant

	array      *Array
	multiplier []float32
}


//func newScalar(name string) *Field {
//	return newField(name, 1, nil)
//}

//func newField(name string, nComp int, size3D []int) *Field {
//	f := new(Field)
//	f.Init(name, nComp, size3D)
//	return f
//}
//
//
//
//func newVector(name string) *Field {
//	return newField(name, 3, nil)
//}
//
//func newScalarField(name string, size3D []int) *Field {
//	return newField(name, 1, size3D)
//}
//
//func newVectorField(name string, size3D []int) *Field {
//	return newField(name, 3, size3D)
//}
//
//// Initiates a field with nComp components and array size size3D.
//// When size3D == nil, the field is space-independent (homogeneous).
//func (f *Field) Init(name string, nComp int, size3D []int) {
//	Assert(nComp > 0 && nComp <= FIELD_MAX_COMP)
//	Assert(size3D == nil || len(size3D) == 3)
//
//	f.array.Free()
//	if size3D != nil {
//		f.array.Init(nComp, size3D)
//	}
//
//	for i := range f._multiplier {
//		f._multiplier[i] = 1
//	}
//	f.multiplier = f._multiplier[:nComp]
//
//	f.name = name
//}
//
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
